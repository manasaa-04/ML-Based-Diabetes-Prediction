"""Rule-based health assessment and generic diet/lifestyle suggestions. Not medical advice."""
DISCLAIMER = ("This tool is an educational decision-support aid, not a medical device or diagnosis. "
              "Consult a qualified clinician or registered dietitian before changing diet or treatment.")


def assess(row: dict) -> list:
    """Return (label, status, note) tuples; status in good/watch/high."""
    out = []
    g, bmi, bp, age = row.get("Glucose"), row.get("BMI"), row.get("BloodPressure"), row.get("Age")
    if g:
        if g < 100: out.append(("Glucose", "good", f"{g} mg/dL, within the normal fasting range if fasting (<100)."))
        elif g < 126: out.append(("Glucose", "watch", f"{g} mg/dL is in the pre-diabetic range (100-125) if fasting."))
        else: out.append(("Glucose", "high", f"{g} mg/dL is in the diabetic range (>=126) if fasting."))
    if bmi:
        if bmi < 18.5: out.append(("BMI", "watch", f"{bmi:.1f}: underweight."))
        elif bmi < 25: out.append(("BMI", "good", f"{bmi:.1f}: healthy weight."))
        elif bmi < 30: out.append(("BMI", "watch", f"{bmi:.1f}: overweight."))
        else: out.append(("BMI", "high", f"{bmi:.1f}: obese range."))
    if bp:
        if bp < 80: out.append(("Diastolic BP", "good", f"{bp} mmHg: normal."))
        elif bp < 90: out.append(("Diastolic BP", "watch", f"{bp} mmHg: elevated."))
        else: out.append(("Diastolic BP", "high", f"{bp} mmHg: high."))
    if age and age >= 45:
        out.append(("Age", "watch", "Risk rises after 45; regular screening is recommended."))
    return out


def suggestions(row: dict, risk: float) -> dict:
    diet = ["Fill half your plate with non-starchy vegetables; choose whole grains over refined carbs.",
            "Prefer water or unsweetened drinks; avoid sugary beverages and juices.",
            "Include lean protein and healthy fats (legumes, fish, nuts, olive oil)."]
    lifestyle = ["Aim for 150 minutes/week of moderate activity (e.g. brisk walking).",
                 "Sleep 7-9 hours and manage stress."]
    if (row.get("Glucose") or 0) >= 100:
        diet.append("Spread carbohydrates evenly across meals and pair them with protein or fibre.")
    if (row.get("BMI") or 0) >= 25:
        lifestyle.append("A 5-7% weight loss substantially lowers type 2 diabetes risk.")
    if risk >= 0.5:
        lifestyle.append("Book a fasting glucose / HbA1c test with a clinician soon.")
    else:
        lifestyle.append("Repeat screening at least yearly, sooner if symptoms appear.")
    return {"diet": diet, "lifestyle": lifestyle, "disclaimer": DISCLAIMER}
