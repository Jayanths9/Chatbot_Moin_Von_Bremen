import os, logging
from typing import Optional
from haystack.schema import Answer

def print_answers(results: dict, details: str = "all", max_text_len: Optional[int] = None):
    """
    Utility function to print results of Haystack pipelines
    :param results: Results that the pipeline returned.
    :param details: Defines the level of details to print. Possible values: minimum, medium, all.
    :param max_text_len: Specifies the maximum allowed length for a text field. If you don't want to shorten the text, set this value to None.
    :return: None
    """
    # Defines the fields to keep in the Answer for each detail level
    fields_to_keep_by_level = {
        "minimum": {Answer: ["answer"]},
        "medium": {Answer: ["answer", "context", "score"]},
    }

    if not "answers" in results.keys():
        raise ValueError(
            "The results object does not seem to come from a Reader: "
            f"it does not contain the 'answers' key, but only: {results.keys()}.  "
            "Try print_documents or print_questions."
        )

    queries = []
    if "query" in results.keys():  # results came from a `run` call
        queries = [results["query"]]
        answers_lists = [results["answers"]]
    elif "queries" in results.keys():  # results came from a `run_batch` call
        queries = results["queries"]
        answers_lists = results["answers"]

    for query_idx, answers in enumerate(answers_lists):
        # Filter the results by detail level
        filtered_answers = []
        if details in fields_to_keep_by_level.keys():
            for ans in answers:
                filtered_ans = {
                    field: getattr(ans, field)
                    for field in fields_to_keep_by_level[details][type(ans)]
                    if getattr(ans, field) is not None
                }
                filtered_answers.append(filtered_ans)
        elif details == "all":
            filtered_answers = answers
        else:
            valid_values = ", ".join(fields_to_keep_by_level.keys()) + " and 'all'"
            logging.warn("print_answers received details='%s', which was not understood. ", details)
            logging.warn("Valid values are %s. Using 'all'.", valid_values)
            filtered_answers = answers

        # Shorten long text fields
        if max_text_len is not None:
            for ans in answers:
                if getattr(ans, "context") and len(ans.context) > max_text_len:
                    ans.context = ans.context[:max_text_len] + "..."

        if len(answers) > 0:
            print(answers[0].answer)