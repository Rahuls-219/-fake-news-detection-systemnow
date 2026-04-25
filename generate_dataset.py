"""
generate_dataset.py (v2)
-------------------------
Creates a richer, more balanced synthetic training dataset.
Doubles the vocabulary diversity to improve model generalization.

Run: python generate_dataset.py
"""

import pandas as pd
import random

random.seed(42)

# ── Fake news templates ───────────────────────────────────────────────────────
FAKE_TEMPLATES = [
    # Conspiracy
    "BREAKING: {entity} secretly {action} to control the {target}, whistleblower reveals",
    "EXPOSED: {entity} has been {action} for {duration} — mainstream media silent",
    "SHOCKING TRUTH: {entity} are {action} to {outcome} — share before deleted",
    "ALERT: {entity} putting {substance} in {target} to {outcome}, leaked documents show",
    "Scientists HATE this: {entity} discovers {miracle} that cures {disease} overnight",
    "URGENT: New {law} will allow {entity} to {action} starting next month",
    "Doctors BANNED from revealing: {miracle} eliminates {disease} in {duration}",
    "NASA/Government covering up: {entity} found on {location}, truth suppressed",
    "WAKE UP: {entity} using {technology} to {action} every citizen secretly",
    "BREAKING: {celebrity} confirms {conspiracy_claim} in explosive interview",
    # Sensational medical/health
    "Big Pharma hiding {miracle} that {outcome} better than any medication ever made",
    "This {food} banned in {country} because it {miracle_claim} — FDA won't tell you why",
    "Ancient {entity} remedy {action} cancer cells in {duration}, suppressed for {duration}",
    # Political
    "LEAKED: {politician} secretly meeting with {entity} to plan {outcome}",
    "BREAKING: {politician} caught {action} on hidden camera, media silent",
    "New evidence proves {historical_event} was staged by {entity}, documents released",
]

FAKE_FILLS = {
    "entity":        ["government", "global elites", "Big Pharma", "tech giants", "scientists", "military", "the deep state", "WHO", "Bill Gates"],
    "action":        ["poisoning water supplies", "tracking citizens", "controlling weather", "suppressing cures", "staging events", "monitoring thoughts", "eliminating cash"],
    "target":        ["population", "food supply", "drinking water", "economy", "media", "children", "public"],
    "duration":      ["50 years", "decades", "centuries", "10 years", "your entire life"],
    "outcome":       ["reduce world population", "control your mind", "steal your money", "spy on families", "dominate the globe"],
    "substance":     ["microchips", "chemicals", "mind-control drugs", "fluoride", "estrogen", "nanoparticles"],
    "miracle":       ["plant compound", "ancient herb", "natural remedy", "fruit extract", "simple trick", "one weird cure"],
    "disease":       ["cancer", "diabetes", "Alzheimer's", "HIV", "heart disease", "all diseases"],
    "law":           ["executive order", "secret bill", "emergency mandate", "hidden regulation"],
    "technology":    ["5G towers", "smart devices", "vaccines", "chemtrails", "satellites"],
    "celebrity":     ["Famous actor", "A-list celebrity", "Top scientist", "Whistleblower insider", "Former government official"],
    "conspiracy_claim": ["vaccines contain tracking chips", "the moon landing was faked", "COVID was man-made", "climate change is a hoax"],
    "food":          ["broccoli extract", "apricot seeds", "baking soda", "cannabis oil", "turmeric"],
    "country":       ["Europe", "Canada", "Australia", "31 countries"],
    "miracle_claim": ["kills cancer cells instantly", "reverses aging completely", "cures autism", "eliminates all viruses"],
    "politician":    ["Senator", "The President", "Governor", "Prime Minister"],
    "historical_event": ["the moon landing", "9/11", "the election", "the pandemic", "the shooting"],
    "location":      ["Mars", "Antarctica", "the moon", "Area 51", "the ocean floor"],
}

# ── Real news templates ───────────────────────────────────────────────────────
REAL_TEMPLATES = [
    "{institution} publishes {study_type} linking {factor} to {health_outcome} in {population}",
    "{government_body} approves {amount} funding for {infrastructure} over {duration}",
    "Researchers at {university} develop new {technology} that improves {outcome} by {percent}",
    "{company} reports quarterly {metric} of {amount}, {comparison} analyst forecasts",
    "{international_body} summit concludes with agreement to {goal} by {year}",
    "New {legislation} introduced to address {policy_issue} amid growing concerns",
    "{central_bank} holds interest rates steady at {rate}%, citing {economic_factor}",
    "Scientists confirm {scientific_finding}, study published in {journal}",
    "{government_agency} releases data showing {trend} in {timeframe}",
    "Phase {phase} clinical trials begin for {drug_type} targeting {disease}",
    "{country} reports {percent}% increase in {metric} following new {policy}",
    "{company} faces {legal_action} from {regulator} over {issue}",
    "Annual {report_type} shows {finding} in {sector} over past {timeframe}",
    "New {infrastructure} project approved after {duration} of environmental review",
    "{health_authority} updates guidelines on {health_topic} based on {evidence_type}",
]

REAL_FILLS = {
    "institution":     ["Johns Hopkins University", "The CDC", "WHO", "MIT researchers", "Oxford scientists", "The NIH", "Stanford researchers"],
    "study_type":      ["peer-reviewed study", "longitudinal research", "meta-analysis", "clinical trial", "observational study"],
    "factor":          ["sedentary lifestyle", "Mediterranean diet", "air pollution", "sleep deprivation", "moderate exercise", "social connection"],
    "health_outcome":  ["cardiovascular risk", "cognitive decline", "life expectancy", "cancer rates", "mental health outcomes"],
    "population":      ["adults over 50", "children aged 5-12", "urban populations", "elderly patients", "women over 40"],
    "government_body": ["Congress", "The Senate", "City Council", "Parliament", "The European Commission"],
    "amount":          ["$1.2 trillion", "$500 million", "€2 billion", "$850 million", "$3.5 billion"],
    "infrastructure":  ["roads and bridges", "broadband internet", "public transport", "green energy", "affordable housing", "water systems"],
    "duration":        ["five years", "the next decade", "three years", "2025-2030", "ten years"],
    "university":      ["MIT", "Stanford", "Oxford", "Harvard", "Cambridge", "ETH Zurich"],
    "technology":      ["battery technology", "water purification method", "drug compound", "AI diagnostic tool", "solar panel"],
    "outcome":         ["energy storage", "charging speed", "detection accuracy", "treatment efficacy", "crop yields"],
    "percent":         ["15 percent", "23 percent", "40 percent", "12 percent", "60 percent", "7 percent"],
    "company":         ["Apple", "Microsoft", "Amazon", "Toyota", "Pfizer", "Tesla", "Google"],
    "metric":          ["earnings", "revenue", "profit", "sales", "output", "exports"],
    "comparison":      ["exceeding", "meeting", "slightly below", "in line with", "surpassing"],
    "international_body": ["G7", "G20", "UN Climate", "WTO", "NATO", "ASEAN"],
    "goal":            ["reduce carbon emissions by 30%", "protect 30% of ocean areas", "eliminate extreme poverty", "expand vaccine access globally"],
    "year":            ["2030", "2035", "2040", "2028"],
    "legislation":     ["data privacy bill", "climate act", "infrastructure package", "health reform bill", "antitrust legislation"],
    "policy_issue":    ["data privacy", "climate change", "housing affordability", "food safety", "workplace safety"],
    "central_bank":    ["The Federal Reserve", "The ECB", "The Bank of England", "The Bank of Japan"],
    "rate":            ["5.25", "4.75", "3.50", "0.10", "4.00"],
    "economic_factor": ["persistent inflation", "strong labor market", "global uncertainty", "cooling consumer demand"],
    "scientific_finding": ["ozone layer recovery is on track", "deep-sea species discovered near hydrothermal vents", "gravitational waves detected from neutron star merger"],
    "journal":         ["Nature", "The Lancet", "Science", "NEJM", "JAMA", "Cell"],
    "government_agency": ["The Labor Department", "EPA", "FDA", "CBO", "BLS"],
    "trend":           ["significant decline in workplace injuries", "steady recovery in manufacturing", "record voter turnout", "drop in smoking rates"],
    "timeframe":       ["the past year", "Q3 2024", "the past decade", "the last six months"],
    "phase":           ["2", "3", "1b"],
    "drug_type":       ["experimental gene therapy", "monoclonal antibody treatment", "small molecule inhibitor", "mRNA vaccine"],
    "disease":         ["Alzheimer's disease", "antibiotic-resistant bacteria", "Type 2 diabetes", "pancreatic cancer"],
    "country":         ["Germany", "South Korea", "Canada", "Brazil", "Australia", "Japan"],
    "policy":          ["safety regulations", "investment program", "tax incentive", "environmental standard"],
    "legal_action":    ["antitrust investigation", "regulatory review", "class-action lawsuit", "compliance audit"],
    "regulator":       ["European Commission", "FTC", "SEC", "FDA", "DOJ"],
    "issue":           ["data privacy violations", "anti-competitive practices", "market manipulation", "safety standards"],
    "report_type":     ["industry report", "government audit", "sustainability report", "economic review"],
    "finding":         ["significant improvement", "steady growth", "notable decline", "record performance"],
    "sector":          ["renewable energy", "healthcare", "technology", "agriculture", "manufacturing"],
    "health_authority": ["WHO", "CDC", "NHS", "European Medicines Agency"],
    "health_topic":    ["antibiotic use", "COVID booster schedules", "screen time for children", "dietary guidelines", "exercise recommendations"],
    "evidence_type":   ["latest clinical evidence", "new research findings", "updated meta-analyses", "real-world data"],
}


def _fill_template(template: str, fills: dict) -> str:
    result = template
    for key, options in fills.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    return result


def generate_records(n_fake=300, n_real=300) -> list[dict]:
    records = []

    # Template-generated
    for _ in range(n_fake):
        tpl = random.choice(FAKE_TEMPLATES)
        records.append({"text": _fill_template(tpl, FAKE_FILLS), "label": "FAKE"})

    for _ in range(n_real):
        tpl = random.choice(REAL_TEMPLATES)
        records.append({"text": _fill_template(tpl, REAL_FILLS), "label": "REAL"})

    # Add curated anchor examples (from original generate_dataset.py)
    FAKE_ANCHORS = [
        "BREAKING: Scientists discover miracle cure for all diseases hidden by Big Pharma for decades",
        "SHOCKING: Government secretly putting mind control chemicals in drinking water, whistleblower reveals",
        "ALERT: 5G towers are being used to spread the virus, experts are being silenced",
        "EXPOSED: The moon landing was faked in a Hollywood studio, new documents prove",
        "Doctors HATE this one weird trick that cures cancer in 24 hours",
        "SHARE BEFORE DELETED: The truth about fluoride in water will shock you",
        "Scientists confirm that the sun is actually getting colder and ice age is coming soon",
        "EXPOSED: All major elections for the past 20 years have been rigged by elites",
        "They are putting birth control in fast food to reduce world population, whistleblower reveals",
    ]
    REAL_ANCHORS = [
        "Federal Reserve raises interest rates by 25 basis points amid ongoing inflation concerns",
        "NASA's James Webb Space Telescope captures detailed images of distant galaxies",
        "World Health Organization releases updated guidelines on antibiotic resistance prevention",
        "Study finds Mediterranean diet associated with lower risk of heart disease in older adults",
        "Scientists sequence genome of rare endangered species to assist conservation efforts",
        "Supreme Court hears arguments in landmark case involving digital privacy rights",
        "University researchers develop new battery technology that charges 40 percent faster",
        "Global summit on biodiversity concludes with agreement to protect 30 percent of land by 2030",
        "Electric vehicle sales surpass diesel car sales for the first time in European market",
        "Researchers achieve milestone in quantum computing reaching new error correction threshold",
    ]

    for t in FAKE_ANCHORS:
        records.append({"text": t, "label": "FAKE"})
    for t in REAL_ANCHORS:
        records.append({"text": t, "label": "REAL"})

    return records


def main():
    import os
    os.makedirs("data", exist_ok=True)

    records = generate_records(n_fake=300, n_real=300)
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/news_dataset.csv", index=False)

    fake_n = (df.label == "FAKE").sum()
    real_n = (df.label == "REAL").sum()
    print(f"[✓] Dataset created: {len(df)} samples  (FAKE: {fake_n}  REAL: {real_n})")
    print(f"    Saved → data/news_dataset.csv")


if __name__ == "__main__":
    main()
