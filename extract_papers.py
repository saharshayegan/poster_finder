# python

# Input: an .html file of a conference schedule webpage, Example title in the page, example abstract in the page
# Output: list of all of the poster titles and their abstract, filtered by LLM for relevance to specified topics
#


from bs4 import BeautifulSoup
import re
import json
from openai import OpenAI
from typing import List, Dict
import os
import argparse
from pathlib import Path


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
    chunk_size: int = 20,
    output_prefix: str = "output",
) -> List[Dict[str, str]]:
    """
    Use an LLM to filter papers related to specific topics.
    Automatically chunks papers into smaller batches to avoid context length limits.

    Args:
        papers: List of dictionaries with 'title' and 'abstract' keys
        topics_of_interest: List of topics to filter for
        api_key: OpenAI API key (if None, will try to use environment variable)
        chunk_size: Number of papers to process in each batch
        output_prefix: Prefix for output files (used for debug file naming)

    Returns:
        List of papers that are relevant to the specified topics
    """
    if not papers:
        return []

    # Set up OpenAI client
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    all_relevant_papers = []
    failed_responses = []

    # Process papers in chunks
    for chunk_start in range(0, len(papers), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(papers))
        chunk_papers = papers[chunk_start:chunk_end]

        print(f"Processing papers {chunk_start + 1}-{chunk_end} of {len(papers)}...")

        # Create the prompt for this chunk
        prompt = f"""
You are an expert AI researcher. Please analyze the following papers and identify which ones are related to ANY of these topics if any:
{', '.join(topics_of_interest)}

Papers to analyze:
"""

        # Add papers to the prompt (only this chunk)
        for i, paper in enumerate(chunk_papers, 1):
            prompt += f"\n{i}. Title: {paper['title']}\n"
            prompt += f"   Abstract: {paper['abstract']}\n"

        prompt += f"""
Please respond with ONLY a JSON array containing the indices of papers that are relevant to the topics: {', '.join(topics_of_interest)}

Example response format: [1, 3, 5]
If no papers are relevant, respond with: []
"""

        try:
            # Make API call to OpenAI using new v1.0+ interface
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert AI researcher. Your task is to identify papers related to these topics: {', '.join(topics_of_interest)}. Respond with ONLY a JSON array of paper indices (numbers). Do not include explanations or other text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,  # Increased to handle longer inputs with abstracts
            )

            # Parse the response
            response_text = response.choices[0].message.content

            # Try to extract JSON from the response
            try:
                # Clean the response to extract just the JSON array
                response_text = response_text.strip()

                # Find the JSON array in the response
                if response_text.startswith("[") and response_text.endswith("]"):
                    # Response is already a clean JSON array
                    relevant_indices = json.loads(response_text)
                else:
                    # Try to find the JSON array within the response
                    array_start = response_text.find("[")
                    array_end = response_text.rfind("]")
                    if array_start != -1 and array_end != -1:
                        json_str = response_text[array_start : array_end + 1]
                        relevant_indices = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError(
                            "No valid JSON array found in response", response_text, 0
                        )

                # Validate that we have a list of integers
                if not isinstance(relevant_indices, list):
                    raise ValueError("Response is not a list")

                # Process the relevant papers
                for paper_index in relevant_indices:
                    if isinstance(paper_index, int) and 1 <= paper_index <= len(
                        chunk_papers
                    ):
                        # Convert to 0-based index
                        zero_based_index = paper_index - 1
                        original_paper = chunk_papers[zero_based_index].copy()
                        # original_paper["llm_explanation"] = (
                        #     f"Selected as relevant to: {', '.join(topics_of_interest)}"
                        # )
                        all_relevant_papers.append(original_paper)

                        # Print the selected paper info immediately
                        print(f"  ✓ Paper {paper_index}: {original_paper['title']}")
                        if "position" in original_paper:
                            print(f"    Position: {original_paper['position']}")
                    else:
                        print(
                            f"  ⚠ Invalid paper index: {paper_index} (valid range: 1-{len(chunk_papers)})"
                        )

            except (json.JSONDecodeError, ValueError) as e:
                print(
                    f"Warning: Could not parse LLM response as JSON for chunk {chunk_start + 1}-{chunk_end}."
                )
                print("Raw LLM response:")
                print("-" * 50)
                print(response_text)
                print("-" * 50)
                print("Skipping this chunk.")

                # Store failed response for file output
                failed_responses.append(
                    {
                        "chunk_range": f"{chunk_start + 1}-{chunk_end}",
                        "prompt": prompt,
                        "response": response_text,
                    }
                )
                continue

        except Exception as e:
            if "context_length_exceeded" in str(e):
                print(
                    f"Context length exceeded for chunk {chunk_start + 1}-{chunk_end}. Trying smaller chunk size..."
                )
                # Recursively try with smaller chunk size
                smaller_chunk_size = max(1, chunk_size // 2)
                smaller_chunk_results = filter_papers_with_llm(
                    chunk_papers,
                    topics_of_interest,
                    api_key,
                    smaller_chunk_size,
                    output_prefix,
                )
                all_relevant_papers.extend(smaller_chunk_results)
            else:
                print(
                    f"Error calling LLM API for chunk {chunk_start + 1}-{chunk_end}: {e}"
                )
                print(f"Skipping chunk {chunk_start + 1}-{chunk_end}")
                continue

    # Save failed responses to debug file if any occurred
    if failed_responses:
        debug_file = f"{output_prefix}_llm_debug_responses.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("LLM RESPONSES THAT FAILED JSON PARSING\n")
            f.write("=" * 80 + "\n\n")
            for failure in failed_responses:
                f.write(f"CHUNK: {failure['chunk_range']}\n")
                f.write("-" * 40 + "\n")
                f.write("PROMPT SENT:\n")
                f.write(failure["prompt"])
                f.write("\n" + "-" * 40 + "\n")
                f.write("RAW RESPONSE:\n")
                f.write(failure["response"])
                f.write("\n" + "=" * 80 + "\n\n")
        print(f"Debug: Failed LLM responses saved to {debug_file}")

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
    relevant_papers = filter_papers_with_llm(
        pairs, args.topics, output_prefix=output_prefix
    )

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
        # if "llm_explanation" in paper:
        #     print(f"   LLM Explanation: {paper['llm_explanation']}")

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
            # if "llm_explanation" in paper:
            #     output_file.write(f"LLM Explanation: {paper['llm_explanation']}\n")
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
