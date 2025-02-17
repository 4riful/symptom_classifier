"""
fallback.py
------------
Provides a naive keyword-based fallback classification when the model cannot confidently
predict a department. This is a last-resort approach to ensure at least some department
is suggested, even if the model's prediction was below a reasonable confidence threshold.
"""

def fallback_classification(text: str) -> str:
    """
    If the model can't find any confident department, we do naive checks on the text.
    This is not a medically accurate approach; it's purely a demonstration of how you
    might implement a backup rule-based system.

    :param text: The input symptom text from the user.
    :return: A string representing the fallback department.
    """
    lower_text = text.lower()

    # Very naive rules for demonstration
    if "rash" in lower_text or "itch" in lower_text:
        return "Dermatology"
    if "heart" in lower_text or "chest pain" in lower_text:
        return "Cardiology"
    if "headache" in lower_text or "migraine" in lower_text:
        return "Neurology"
    if "bone" in lower_text or "fracture" in lower_text:
        return "Orthopedics"
    if "child" in lower_text or "kid" in lower_text:
        return "Pediatrics"
    if "lung" in lower_text or "breathing" in lower_text:
        return "Pulmonology"

    # Default fallback
    return "General Medicine"
