import argparse
import json
import os
import os.path as osp
import re
import shutil
import traceback
import unicodedata

from ai_scientist.perform_icbinb_writeup import (
    compile_latex,
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
    check_page_limit,
    get_reflection_page_info,
)
from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
    AVAILABLE_LLMS,
)
from ai_scientist.tools.semantic_scholar import search_for_papers
from ai_scientist.perform_vlm_review import (
    generate_vlm_img_review,
    perform_imgs_cap_ref_review,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client


def _remove_accents_and_clean(s):
    nfkd_form = unicodedata.normalize("NFKD", s)
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    ascii_str = ascii_str.lower()
    return ascii_str


# ---------------------------------------------------------------------------
# Survey-specific citation system message
# ---------------------------------------------------------------------------
survey_citation_system_msg = """You are writing a comprehensive survey paper targeting a top-tier academic journal. \
Your goal is to gather diverse references across ALL topic areas covered by the survey. \
Aim for 200+ unique references spanning different sub-areas, methods, datasets, and time periods. \
Do NOT stop early — a survey needs much broader citation coverage than a regular paper.

This phase focuses on collecting references and annotating them to be integrated later.
Collected citations will be added to a references.bib file.

Reasons to reference papers include:
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts: Provide citations when discussing specific theories or concepts.
3. Datasets, models, and optimizers: Cite the creators of datasets, models, and optimizers.
4. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
5. Highlighting Research Gaps: Cite previous research when pointing out gaps your study addresses.
6. Using Established Methods: Cite the creators of methodologies you employ.
7. Supporting Arguments: Cite sources that back up your conclusions and arguments.
8. Suggesting Future Research: Reference studies related to proposed future research directions.

Ensure sufficient cites will be collected for all of these categories, and no categories are missed.
You will be given access to the Semantic Scholar API; only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.
You will have {total_rounds} rounds to add to the references but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

_citation_first_prompt_template = """Round {current_round}/{total_rounds}:

You planned and executed the following idea:
```markdown
{Idea}
```

You produced the following report:
```markdown
{report}
```

Your current list of citations is:
```
{citations}
```

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason and identify which citations are missing.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": The purpose of the desired citation and a brief description of what you are looking for.
- "Query": The search query to find the paper (e.g., attention is all you need).
This JSON will be automatically parsed, so ensure the format is precise."""

_citation_second_prompt_template = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, briefly reason over the search results and identify which citation(s) best fit your paper.
If none are appropriate or would contribute significantly to the write-up, add "Do not add any" to your thoughts.
Do not select papers that are already in the `references.bib` file, or if the same citation exists under a different name.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of integer indices for the selected papers, for example [0, 1]. Do not use quotes for the indices, e.g. "['0', '1']" is invalid.
- "Description": Update the previous description of the citation(s) with the additional context. This should be a brief description of the work(s), their relevance, and where in a paper these should be cited.
This JSON will be automatically parsed, so ensure the format is precise."""


def _get_survey_citation_addition(
    client, model, context, current_round, total_rounds, idea_text, result_limit=20
):
    report, citations = context
    msg_history = []

    try:
        text, msg_history = get_response_from_llm(
            prompt=_citation_first_prompt_template.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=survey_citation_system_msg.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query, result_limit=result_limit)
    except Exception:
        print("EXCEPTION in _get_survey_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            prompt=_citation_second_prompt_template.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=survey_citation_system_msg.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_indices = json_output.get("Selected", [])
        if isinstance(selected_indices, list) and selected_indices:
            selected_indices = [int(i) for i in selected_indices]
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                if "\n" in bibtex:
                    newline_index = bibtex.find("\n")
                    cite_key_line = bibtex[:newline_index]
                    cite_key_line = _remove_accents_and_clean(cite_key_line)
                    cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
                else:
                    cleaned_bibtexs.append(_remove_accents_and_clean(bibtex))
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in _get_survey_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


def gather_survey_citations(base_folder, num_cite_rounds=100, small_model="gpt-5.4", result_limit=20):
    """
    Survey-specific citation gathering with a broader persona and higher result limits.
    """
    citations_cache_path = osp.join(base_folder, "cached_citations.bib")
    progress_path = osp.join(base_folder, "citations_progress.json")

    # Check for stale / incomplete cache and clear it
    if osp.exists(progress_path):
        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
            if progress.get("status") != "completed":
                print("Clearing incomplete citation cache, re-gathering...")
                if osp.exists(citations_cache_path):
                    os.remove(citations_cache_path)
                os.remove(progress_path)
        except Exception:
            pass

    current_round = 0
    citations_text = ""

    if osp.exists(citations_cache_path) and osp.exists(progress_path):
        try:
            with open(citations_cache_path, "r") as f:
                citations_text = f.read()
            with open(progress_path, "r") as f:
                progress = json.load(f)
                current_round = progress.get("completed_rounds", 0)
            print(f"Resuming citation gathering from round {current_round}")
        except Exception as e:
            print(f"Error loading cached citations: {e}")
            print("Starting fresh")
            current_round = 0
            citations_text = ""

    try:
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="citation_gathering"
        )
        filtered_summaries_str = json.dumps(filtered_summaries, indent=2)

        client, client_model = create_client(small_model)

        for round_idx in range(current_round, num_cite_rounds):
            try:
                context_for_citation = (filtered_summaries_str, citations_text)
                addition, done = _get_survey_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                    result_limit=result_limit,
                )

                if done:
                    with open(citations_cache_path, "w") as f:
                        f.write(citations_text)
                    with open(progress_path, "w") as f:
                        json.dump(
                            {"completed_rounds": round_idx + 1, "status": "completed"},
                            f,
                        )
                    break

                if addition is not None:
                    title_match = re.search(r"title\s*=\s*[\{\"](.*?)[\}\"]", addition, re.IGNORECASE)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r"title\s*=\s*[\{\"](.*?)[\}\"]", citations_text, re.IGNORECASE
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            citations_text += "\n" + addition
                            with open(citations_cache_path, "w") as f:
                                f.write(citations_text)
                            with open(progress_path, "w") as f:
                                json.dump(
                                    {
                                        "completed_rounds": round_idx + 1,
                                        "status": "in_progress",
                                    },
                                    f,
                                )

            except Exception as e:
                print(f"Error in citation round {round_idx}: {e}")
                print(traceback.format_exc())
                with open(citations_cache_path, "w") as f:
                    f.write(citations_text)
                with open(progress_path, "w") as f:
                    json.dump({"completed_rounds": round_idx, "status": "error"}, f)
                continue

        return citations_text if citations_text else None

    except Exception:
        print("EXCEPTION in gather_survey_citations:")
        print(traceback.format_exc())
        return citations_text if citations_text else None


# Survey-specific system message for IEEE TPAMI double-column format
writeup_system_message_template = """You are an expert academic researcher writing a comprehensive survey paper for IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
Your survey must be scientifically accurate, objective, and provide insightful analysis -- not merely list or summarize existing works.
The paper uses IEEE Transactions double-column format and should be {page_limit} pages (excluding references).
DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT (before references).

Key requirements:
- Use \\cite{{}} for all references from references.bib. Cite extensively — aim for 200+ references. Every section should have at minimum 5 unique citations. When meeting page limits, compress prose paragraphs but NEVER reduce citation density. Never cut citations to save space.
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

- **Body sections** (organized by the survey's taxonomy):
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
  - Discuss emerging opportunities.

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
5. Use figures selectively. Figures that illustrate the survey's technical content (method comparisons, performance trends, taxonomy diagrams) should be prioritized. Figures about data-collection methodology should only appear in the Statistical Overview section. Do not exceed 2 figures per section on average. It is acceptable to omit figures that don't add substantive value.
6. Citations from references.bib are used extensively throughout.
7. For each method discussed, extract and report specific numerical results (mAP, IoU, F1, accuracy) from the summaries. Build comparison tables with columns: Method | Year | Dataset | Metric | Score. If a value is unavailable, use `--` and note `(not reported)`. Do NOT invent numbers.
8. Every section must contain at minimum: (a) an opening paragraph explaining scope, (b) a comparison table or taxonomy with at least 3 entries, (c) a synthesis paragraph on trends and open problems. A section with fewer than 300 words is incomplete.

Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup for a TPAMI survey.
Make sure to use the citations from the references.bib file.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""

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
        with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
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
            progress_path = osp.join(base_folder, "citations_progress.json")

            # Check for stale / incomplete cache and clear it
            if osp.exists(progress_path):
                try:
                    with open(progress_path, "r") as f:
                        progress = json.load(f)
                    if progress.get("status") != "completed":
                        print("Clearing incomplete citation cache, re-gathering...")
                        if osp.exists(citations_cache_path):
                            os.remove(citations_cache_path)
                        os.remove(progress_path)
                except Exception:
                    pass

            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them with survey-specific persona
            if not citations_text:
                num_cite_rounds = max(num_cite_rounds, 100)
                citations_text = gather_survey_citations(
                    base_folder, num_cite_rounds, small_model, result_limit=20
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text:
            with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
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
        # Survey writeup needs larger output tokens than the safe default (16384)
        # GPT-5.x supports up to 128000 via max_completion_tokens
        import ai_scientist.llm as llm_module
        original_max_tokens = llm_module.MAX_NUM_TOKENS
        if "gpt-5" in big_model:
            llm_module.MAX_NUM_TOKENS = 128000
        try:
            big_client, big_client_model = create_client(big_model)
            with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
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
                with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
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
6) For unused figures, decide if they add substantive value to a specific section. If not, it is acceptable to omit them. Unused figures: {sorted(unused_figs)}
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
10) Identify any section with fewer than 5 unique \\cite{{}} references. Those sections need more citations.
11) Identify any section with fewer than 200 words. Those sections need expansion.

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
                with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
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
        finally:
            llm_module.MAX_NUM_TOKENS = original_max_tokens

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
