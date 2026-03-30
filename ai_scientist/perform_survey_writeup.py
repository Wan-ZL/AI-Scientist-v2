import argparse
import json
import os
import os.path as osp
import re
import shutil
import traceback

from ai_scientist.perform_icbinb_writeup import (
    compile_latex,
    gather_citations,
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
    check_page_limit,
    get_reflection_page_info,
)
from ai_scientist.llm import (
    get_response_from_llm,
    create_client,
    AVAILABLE_LLMS,
)
from ai_scientist.perform_vlm_review import (
    generate_vlm_img_review,
    perform_imgs_cap_ref_review,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client


# Survey-specific system message for IEEE TPAMI double-column format
writeup_system_message_template = """You are an expert academic researcher writing a comprehensive survey paper for IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
Your survey must be scientifically accurate, objective, and provide insightful analysis -- not merely list or summarize existing works.
The paper uses IEEE Transactions double-column format and should be {page_limit} pages (excluding references).
DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT (before references).

Key requirements:
- Use \\cite{{}} for all references from references.bib. Cite 200+ papers where appropriate.
- Do not change the overall style mandated by IEEE. Keep the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.
- Provide quantitative analysis with comparison tables and statistical charts wherever possible.
- Go beyond listing papers: synthesize trends, identify patterns, and offer critical analysis.

Here are tips for each section of the survey:

- **Title**:
  - Should clearly convey the survey scope and domain.
  - Keep it concise but descriptive (under 2 lines).

- **Abstract**:
  - Summarize the survey scope, methodology, key findings, and contributions.
  - State the number of papers reviewed and the time span covered.
  - One continuous paragraph.

- **Introduction**:
  - Motivate the survey: why is this domain important and timely?
  - Summarize the survey's scope, methodology (how papers were collected/filtered), and structure.
  - Highlight key contributions and insights the survey provides beyond existing surveys.

- **Background and Taxonomy**:
  - Define key terminology and problem formulations.
  - Present the proposed taxonomy with a clear hierarchy.
  - Include a taxonomy figure if available.

- **Statistical Overview**:
  - Quantitative analysis of the surveyed literature: publication trends over time, venue distribution, method popularity.
  - Use bar charts, line plots, and tables to present statistics.
  - Discuss what the trends reveal about the field's evolution.

- **Per-method sections** (Symbol Detection, Layout Analysis, Text Recognition, Topology, Floor Plans, Engineering Diagrams, Foundation Models, VLM, High-Resolution):
  - For each section, provide a structured review of approaches in that area.
  - Include comparison tables with methods, datasets used, and reported metrics.
  - Discuss strengths, limitations, and evolution of approaches.
  - Cross-reference related sections where methods overlap.

- **Datasets and Benchmarks**:
  - Comprehensive table of available datasets with key attributes (size, annotation type, domain, availability).
  - Discuss benchmark protocols and evaluation metrics.
  - Identify gaps in dataset coverage.

- **Open Challenges and Future Directions**:
  - Synthesize unsolved problems from across all reviewed areas.
  - Propose concrete future research directions with justification.
  - Discuss emerging opportunities (foundation models, multi-modal approaches, etc.).

- **Conclusion**:
  - Summarize the survey's key findings and contributions.
  - Reiterate the most important open challenges.

Ensure you are always writing good compilable LaTeX code. Common mistakes to fix:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate citations or results not supported by the provided data.

Ensure proper citation usage:
- Always include references within \\begin{{filecontents}}{{references.bib}} ... \\end{{filecontents}}, even if unchanged from the previous round.
- Use citations from the provided references.bib content.
- Every section should have multiple citations.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
"""

# Survey writeup prompt
writeup_prompt = """Your goal is to write a comprehensive survey paper based on the following research plan:

```markdown
{idea_text}
```

We have the following experiment summaries (containing the papers database, statistics, and analysis):
```json
{summaries}
```

We also have a script used to produce the final plots (use this to see how the plots are generated and what names are used in the legend):
```python
{aggregator_code}
```
Please also consider which plots should naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

Your current progress on the LaTeX write-up is:
```latex
{latex_writeup}
```

Produce the final version of the LaTeX survey manuscript now, ensuring:
1. The paper is coherent and provides insightful analysis, not just a listing of papers.
2. All sections from the template are filled with substantive content.
3. Comparison tables are included for each major method category.
4. Statistical figures are referenced and discussed.
5. All available plots are used where appropriate.
6. Citations from references.bib are used extensively throughout.

Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup for a TPAMI survey.
Make sure to use the citations from the references.bib file.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""

# Survey-specific citation system message (used implicitly via gather_citations)
# The gather_citations function uses its own citation prompts from perform_icbinb_writeup.
# For survey mode, we rely on the higher num_cite_rounds to collect more references.


def perform_survey_writeup(
    base_folder,
    citations_text=None,
    no_writing=False,
    num_cite_rounds=50,
    small_model="gpt-5.4",
    big_model="gpt-5.4",
    n_writeup_reflections=5,
    page_limit=30,
):
    pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}.pdf")
    latex_folder = osp.join(base_folder, "latex")

    # Cleanup any previous latex folder and pdf
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    if osp.exists(pdf_file):
        os.remove(pdf_file)

    # Remove any previous reflection PDFs
    for old_pdf in os.listdir(base_folder):
        if old_pdf.endswith(".pdf") and "reflection" in old_pdf:
            os.remove(osp.join(base_folder, old_pdf))

    try:
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries_for_writeup = filter_experiment_summaries(
            exp_summaries, step_name="writeup"
        )
        combined_summaries_str = json.dumps(filtered_summaries_for_writeup, indent=2)

        # Copy TPAMI template (not ICBINB)
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_tpami_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, pdf_file)
            return osp.exists(pdf_file)

        # If no citations provided, try to load from cache first
        if citations_text is None:
            citations_cache_path = osp.join(base_folder, "cached_citations.bib")
            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them (with higher rounds for surveys)
            if not citations_text:
                citations_text = gather_citations(
                    base_folder, num_cite_rounds, small_model
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text:
            with open(writeup_file, "r") as f:
                content = f.read()
            pattern_end = r"\end{filecontents}"
            content = content.replace(pattern_end, f"\n{citations_text}{pattern_end}")
            with open(writeup_file, "w") as f:
                f.write(content)

        # Generate VLM-based descriptions
        try:
            vlm_client, vlm_model = create_vlm_client(small_model)
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        big_model_system_message = writeup_system_message_template.format(
            page_limit=page_limit
        )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        # Try to extract LaTeX from response (with or without language tag)
        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            latex_code_match = re.search(r"```(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            # If no code block, check if response itself looks like LaTeX
            if "\\documentclass" in response or "\\begin{document}" in response:
                updated_latex_code = response.strip()
            else:
                print(f"No LaTeX found in response. First 500 chars: {response[:500]}")
                return False
        else:
            updated_latex_code = latex_code_match.group(1).strip()
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops (more passes for longer survey papers)
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Save PDF with reflection trial number
            reflection_pdf = osp.join(
                base_folder, f"{osp.basename(base_folder)}_reflection{i+1}.pdf"
            )
            print(f"Compiling PDF for reflection {i+1}...")
            compile_latex(latex_folder, reflection_pdf)

            # VLM review of figures
            try:
                review_img_cap_ref = perform_imgs_cap_ref_review(
                    vlm_client, vlm_model, reflection_pdf
                )
            except Exception:
                print("EXCEPTION in VLM image review:")
                print(traceback.format_exc())
                review_img_cap_ref = "No VLM review available."

            # Detect duplicate figures
            try:
                analysis_duplicate_figs = detect_duplicate_figures(
                    vlm_client, vlm_model, reflection_pdf
                )
            except Exception:
                print("EXCEPTION in duplicate figure detection:")
                print(traceback.format_exc())
                analysis_duplicate_figs = "No duplicate analysis available."

            # Get reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

            check_output = os.popen(
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_prompt = f"""
Now let's reflect and identify any issues (including but not limited to):
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.
2) Is the writing clear, well-organized, and does it provide insightful analysis (not just listing papers)?
3) Have we included all relevant details from the summaries without hallucinating?
4) Are comparison tables comprehensive with proper formatting for the double-column layout?
5) Are statistical trends and patterns discussed with quantitative evidence?
6) The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}
7) The following figure references in the LaTeX do not match any actual file: {sorted(invalid_figs)}
{reflection_page_info}
chktex results:
```
{check_output}
```
8) Issues identified in the VLM reviews of the images, their captions, and related text discussions.
VLM reviews:
```
{review_img_cap_ref}
```
9) Duplicate figures detected:
```
{analysis_duplicate_figs}
```

Please provide a revised complete LaTeX in triple backticks, or repeat the same if no changes are needed.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup for a TPAMI survey.
Do not hallucinate any details!
Ensure proper citation usage:
- Always include references within \\begin{{filecontents}}{{references.bib}} ... \\end{{filecontents}}, even if unchanged.
- Use citations from the provided references.bib content.
- Every section should have multiple citations.

If you believe you are done with reflection, simply say: "I am done".
"""

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "\u2019": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(latex_folder, reflection_pdf)
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break

        # Final reflection on page limit
        reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

        final_reflection_prompt = """{reflection_page_info}
USE MINIMAL EDITS TO OPTIMIZE THE PAGE LIMIT USAGE."""
        reflection_response, msg_history = get_response_from_llm(
            prompt=final_reflection_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            msg_history=msg_history[-1:],
            print_debug=False,
        )

        reflection_pdf = osp.join(
            base_folder, f"{osp.basename(base_folder)}_reflection_final_page_limit.pdf"
        )
        print("Compiling PDF for reflection final page limit...")

        reflection_code_match = re.search(
            r"```latex(.*?)```", reflection_response, re.DOTALL
        )
        if reflection_code_match:
            reflected_latex_code = reflection_code_match.group(1).strip()
            with open(writeup_file, "r") as f:
                current_latex = f.read()
            if reflected_latex_code != current_latex:
                final_text = reflected_latex_code
                cleanup_map = {
                    "</end": r"\\end",
                    "</begin": r"\\begin",
                    "\u2019": "'",
                }
                for bad_str, repl_str in cleanup_map.items():
                    final_text = final_text.replace(bad_str, repl_str)
                final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                with open(writeup_file, "w") as fo:
                    fo.write(final_text)

                compile_latex(latex_folder, reflection_pdf)
            else:
                print("No changes in reflection page step.")

        return osp.exists(reflection_pdf)

    except Exception:
        print("EXCEPTION in perform_survey_writeup:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform survey writeup for IEEE TPAMI"
    )
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument(
        "--no-writing", action="store_true", help="Only compile existing LaTeX"
    )
    parser.add_argument("--num-cite-rounds", type=int, default=50)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        choices=AVAILABLE_LLMS,
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="gpt-5.4",
        choices=AVAILABLE_LLMS,
        help="Model to use for final writeup (big model).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=5,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=30,
        help="Target page limit for the main paper (excluding references).",
    )
    args = parser.parse_args()

    try:
        success = perform_survey_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
        )
        if not success:
            print("Survey writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
