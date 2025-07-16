# python

# Input: an .html file of a conference schedule webpage, Example title in the page, example abstract in the page
# Output: list of all of the poster titles and their abstract, filtered by LLM for relevance to specified topics
#
# Usage:
#   python extract_papers.py --input posters.html --topics "LLM agents" "fine-tuning" --output-prefix "icml2025"
#   python extract_papers.py -i conference.html -t "machine learning" "AI" -o "nips2024"
#   python extract_papers.py  # Uses defaults: posters.html and LLM-related topics
#
# Prerequisites:
# 1. Install requirements: pip install -r requirements.txt
# 2. Set OpenAI API key: export OPENAI_API_KEY="your-api-key-here"
#    Or pass it directly to the filter_papers_with_llm function

from bs4 import BeautifulSoup
import re
import json
from openai import OpenAI
from typing import List, Dict
import os
import argparse
from pathlib import Path

example_title = (
    "MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-text Decoding"
)
example_abstract = "Decoding functional magnetic resonance imaging (fMRI) signals into text has been a key challenge in the neuroscience community, with the potential to advance brain-computer interfaces and uncover deeper insights into brain mechanisms. However, existing approaches often struggle with suboptimal predictive performance, limited task variety, and poor generalization across subjects. In response to this, we propose MindLLM, a model designed for subject-agnostic and versatile fMRI-to-text decoding. MindLLM consists of an fMRI encoder and an off-the-shelf LLM. The fMRI encoder employs a neuroscience-informed attention mechanism, which is capable of accommodating subjects with varying input shapes and thus achieves high-performance subject-agnostic decoding. Moreover, we introduce Brain Instruction Tuning (BIT), a novel approach that enhances the model's ability to capture diverse semantic representations from fMRI signals, facilitating more versatile decoding. We evaluate MindLLM on comprehensive fMRI-to-text benchmarks. Results demonstrate that our model outperforms the baselines, improving downstream tasks by $12.0\%$, unseen subject generalization by $24.5\%$, and novel task adaptation by $25.0\%$. Furthermore, the attention patterns in MindLLM provide interpretable insights into its decision-making process."
example_html_chunk = """

                <div class="track-schedule-card">
                    <div None  >
                        <div class="btn-spacer float-end track-pad" style="position:relative; top:-3px">
                            
                            <div class="track-child-bookmark">

    <span id="bookmark-number-1" class="bump20 bookmark-cell fa-lg" title="Add / Remove Bookmark"
          onclick="toggle_bookmark(bookmark_id='bookmark-number-1', event_id='45933', bookmark_event_number='1',
                  alt_bookmark_element_id='');">
        
            
                <i class="fa-bookmark fa-solid solid-bookmark" style="display:none;"></i>
                <i class="fa-bookmark fa-regular regular-bookmark" ></i>
            
        
    </span>


</div>
                            <br>
                            
                                <div class="float-end small">
                                    Poster
                                </div>
                            
                            
                                <div class="small" title="Poster Position">#W-100</div>
                            
                        </div>
                        

                        <h5><strong><a href="/virtual/2025/poster/45933">MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-text Decoding</a></strong></h5>


                        <p class="text-muted">
                            Weikang Qiu &middot; Zheng Huang &middot; Haoyu Hu &middot; Aosong Feng &middot; Yujun Yan &middot; ZHITAO YING
                        </p>

                    </div>
                    <div class="abstract">
                        <p>Decoding functional magnetic resonance imaging (fMRI) signals into text has been a key challenge in the neuroscience community, with the potential to advance brain-computer interfaces and uncover deeper insights into brain mechanisms. However, existing approaches often struggle with suboptimal predictive performance, limited task variety, and poor generalization across subjects. In response to this, we propose MindLLM, a model designed for subject-agnostic and versatile fMRI-to-text decoding. MindLLM consists of an fMRI encoder and an off-the-shelf LLM. The fMRI encoder employs a neuroscience-informed attention mechanism, which is capable of accommodating subjects with varying input shapes and thus achieves high-performance subject-agnostic decoding. Moreover, we introduce Brain Instruction Tuning (BIT), a novel approach that enhances the model's ability to capture diverse semantic representations from fMRI signals, facilitating more versatile decoding. We evaluate MindLLM on comprehensive fMRI-to-text benchmarks. Results demonstrate that our model outperforms the baselines, improving downstream tasks by $12.0\%$, unseen subject generalization by $24.5\%$, and novel task adaptation by $25.0\%$. Furthermore, the attention patterns in MindLLM provide interpretable insights into its decision-making process.</p>
                    </div>

                </div>
                    """


def extract_title_abstract_pairs(html_content):
    """
    Extract title and abstract pairs from HTML content.

    Args:
        html_content (str): HTML content containing poster information

    Returns:
        list: List of dictionaries with 'title' and 'abstract' keys
    """
    soup = BeautifulSoup(html_content, "html.parser")
    pairs = []

    # Find all poster cards
    poster_cards = soup.find_all("div", class_="track-schedule-card")

    for card in poster_cards:
        # Extract title from the h5 strong a tag
        title_element = card.find("h5")
        if title_element:
            title_link = title_element.find("a")
            title = (
                title_link.get_text(strip=True)
                if title_link
                else title_element.get_text(strip=True)
            )
        else:
            title = None

        # Extract abstract from the div with class 'abstract'
        abstract_element = card.find("div", class_="abstract")
        if abstract_element:
            abstract_p = abstract_element.find("p")
            abstract = (
                abstract_p.get_text(strip=True)
                if abstract_p
                else abstract_element.get_text(strip=True)
            )
        else:
            abstract = None

        # Extract poster position from div with title="Poster Position"
        position_element = card.find("div", title="Poster Position")
        position = None
        if position_element:
            position = position_element.get_text(strip=True)

        # Only add pairs where both title and abstract exist
        if title and abstract:
            paper_data = {"title": title, "abstract": abstract}
            if position:
                paper_data["position"] = position
            pairs.append(paper_data)

    return pairs


def filter_papers_with_llm(
    papers: List[Dict[str, str]],
    topics_of_interest: List[str],
    api_key: str = None,
    chunk_size: int = 150,
) -> List[Dict[str, str]]:
    """
    Use an LLM to filter papers related to specific topics.
    Automatically chunks papers into smaller batches to avoid context length limits.

    Args:
        papers: List of dictionaries with 'title' and 'abstract' keys
        topics_of_interest: List of topics to filter for
        api_key: OpenAI API key (if None, will try to use environment variable)
        chunk_size: Number of papers to process in each batch

    Returns:
        List of papers that are relevant to the specified topics
    """
    if not papers:
        return []

    # Set up OpenAI client
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    all_relevant_papers = []

    # Process papers in chunks
    for chunk_start in range(0, len(papers), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(papers))
        chunk_papers = papers[chunk_start:chunk_end]

        print(f"Processing papers {chunk_start + 1}-{chunk_end} of {len(papers)}...")

        # Create the prompt for this chunk
        prompt = f"""
You are an expert AI researcher. Please analyze the following papers and identify which ones are related to ANY of these topics:
{', '.join(topics_of_interest)}

For each paper, respond with either "RELEVANT" or "NOT_RELEVANT" followed by a brief explanation.

Papers to analyze:
"""

        # Add papers to the prompt (only this chunk)
        for i, paper in enumerate(chunk_papers, 1):
            prompt += f"\n{i}. {paper['title']}\n"

        prompt += "\nPlease respond in JSON format with the following structure:\n"
        prompt += '{"results": [{"paper_index": 1, "relevance": "RELEVANT", "explanation": "..."}, ...]}'

        try:
            # Make API call to OpenAI using new v1.0+ interface
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert AI researcher who specializes in identifying papers related to the following topics: {', '.join(topics_of_interest)}.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,  # Reduced from 4000 to leave more room for input
            )

            # Parse the response
            response_text = response.choices[0].message.content

            # Try to extract JSON from the response
            try:
                result_data = json.loads(response_text)
                results = result_data.get("results", [])
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse LLM response as JSON for chunk {chunk_start + 1}-{chunk_end}. Skipping this chunk."
                )
                continue

            # Filter papers based on LLM results for this chunk
            for result in results:
                if result.get("relevance") == "RELEVANT":
                    paper_index = (
                        result.get("paper_index", 1) - 1
                    )  # Convert to 0-based index
                    if 0 <= paper_index < len(chunk_papers):
                        # Get the original paper from the chunk
                        original_paper = chunk_papers[paper_index].copy()
                        original_paper["llm_explanation"] = result.get(
                            "explanation", ""
                        )
                        all_relevant_papers.append(original_paper)

        except Exception as e:
            if "context_length_exceeded" in str(e):
                print(
                    f"Context length exceeded for chunk {chunk_start + 1}-{chunk_end}. Trying smaller chunk size..."
                )
                # Recursively try with smaller chunk size
                smaller_chunk_size = max(1, chunk_size // 2)
                smaller_chunk_results = filter_papers_with_llm(
                    chunk_papers, api_key, smaller_chunk_size
                )
                all_relevant_papers.extend(smaller_chunk_results)
            else:
                print(
                    f"Error calling LLM API for chunk {chunk_start + 1}-{chunk_end}: {e}"
                )
                print(f"Skipping chunk {chunk_start + 1}-{chunk_end}")
                continue

    return all_relevant_papers


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and filter relevant papers from conference HTML"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="posters.html",
        help="Input HTML file (default: posters.html)",
    )

    parser.add_argument(
        "--topics",
        "-t",
        nargs="+",
        default=[
            "LLM agents",
            "fine-tuning LLMs for agents",
            "LLM reasoning",
            "LLM post-training with RL",
            "designing reward for LLM fine-tuning",
        ],
        help="Topics of interest for filtering papers (default: LLM-related topics)",
    )

    parser.add_argument(
        "--output-prefix",
        "-o",
        type=str,
        default=None,
        help="Output file prefix (default: derived from input filename)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Generate output file prefix from input filename if not provided
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        input_path = Path(args.input)
        output_prefix = input_path.stem

    # Read the HTML file
    try:
        with open(args.input, "r", encoding="utf-8") as file:
            html_content = file.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file '{args.input}': {e}")
        return

    # Extract title and abstract pairs
    pairs = extract_title_abstract_pairs(html_content)
    print(f"Found {len(pairs)} title-abstract pairs from {args.input}")

    # Filter papers using LLM
    print(f"Filtering papers with LLM for topics: {', '.join(args.topics)}")
    relevant_papers = filter_papers_with_llm(pairs, args.topics)

    print(f"\nFound {len(relevant_papers)} relevant papers:")
    for i, paper in enumerate(relevant_papers, 1):
        print(f"\n{i}. Title: {paper['title']}")
        if "position" in paper:
            print(f"   Position: {paper['position']}")
        print(
            f"   Abstract: {paper['abstract'][:100]}..."
            if len(paper["abstract"]) > 100
            else f"   Abstract: {paper['abstract']}"
        )
        if "llm_explanation" in paper:
            print(f"   LLM Explanation: {paper['llm_explanation']}")

    # Generate output filenames
    detailed_output_file = f"{output_prefix}_extracted_papers.txt"
    names_output_file = f"{output_prefix}_relevant_paper_names.txt"

    # Save detailed results to file
    with open(detailed_output_file, "w", encoding="utf-8") as output_file:
        output_file.write(f"EXTRACTION FROM: {args.input}\n")
        output_file.write(f"TOPICS SEARCHED: {', '.join(args.topics)}\n")
        output_file.write("=" * 80 + "\n\n")

        output_file.write("ALL EXTRACTED PAPERS:\n")
        output_file.write("=" * 50 + "\n\n")
        for pair in pairs:
            output_file.write(f"Title: {pair['title']}\n")
            if "position" in pair:
                output_file.write(f"Position: {pair['position']}\n")
            output_file.write(f"Abstract: {pair['abstract']}\n\n")

        output_file.write("\n\nRELEVANT PAPERS (LLM FILTERED):\n")
        output_file.write("=" * 50 + "\n\n")
        for paper in relevant_papers:
            output_file.write(f"Title: {paper['title']}\n")
            if "position" in paper:
                output_file.write(f"Position: {paper['position']}\n")
            output_file.write(f"Abstract: {paper['abstract']}\n")
            if "llm_explanation" in paper:
                output_file.write(f"LLM Explanation: {paper['llm_explanation']}\n")
            output_file.write("\n")

    # Save just the relevant paper names to a separate file
    with open(names_output_file, "w", encoding="utf-8") as names_file:
        names_file.write(f"RELEVANT PAPER NAMES FROM: {args.input}\n")
        names_file.write(f"TOPICS SEARCHED: {', '.join(args.topics)}\n")
        names_file.write("=" * 80 + "\n\n")

        for i, paper in enumerate(relevant_papers, 1):
            names_file.write(f"{i}. {paper['title']}")
            if "position" in paper:
                names_file.write(f" ({paper['position']})")
            names_file.write("\n")

    print(f"\nResults saved to:")
    print(f"  - Detailed results: {detailed_output_file}")
    print(f"  - Paper names only: {names_output_file}")


if __name__ == "__main__":
    main()
