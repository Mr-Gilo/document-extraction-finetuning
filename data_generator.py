"""
Synthetic multi-domain incident report generator.
Produces training data for document extraction fine-tuning.

Domains:
  - Health and Safety incident reports
  - IT Security incident reports
  - Field Equipment Maintenance reports

The same extraction schema applies across all domains,
demonstrating a domain-agnostic approach.
"""

import json
import random
import os
from config import DATA_CONFIG

random.seed(DATA_CONFIG["seed"])


# ── Domain: Health and Safety ──────────────────────────────────────────────

HS_LOCATIONS = [
    "Warehouse B, Level 2", "Production Floor A", "Loading Bay 3",
    "Chemical Storage Unit", "Roof Access Area", "Server Room 1",
    "Construction Site, Zone 4", "Laboratory Block C", "Car Park Level 1",
    "Maintenance Workshop"
]

HS_INCIDENT_TYPES = [
    "Slip and fall", "Manual handling injury", "Chemical exposure",
    "Near miss — falling object", "Electrical fault", "Fire alarm activation",
    "Equipment malfunction", "Cuts and lacerations", "Strain injury",
    "Unauthorised access to restricted area"
]

HS_ACTIONS = [
    "Immediate area cordoned off and inspected",
    "First aid administered and incident logged",
    "Equipment taken out of service pending inspection",
    "Safety briefing issued to all staff in affected area",
    "RIDDOR report submitted to HSE",
    "Risk assessment updated and reviewed by safety officer",
    "Corrective maintenance scheduled within 24 hours",
    "Staff member referred to occupational health",
    "Temporary signage installed pending permanent fix",
    "Management review meeting scheduled"
]

HS_NAMES = [
    "J. Thompson", "A. Okafor", "S. Patel", "M. Kowalski",
    "R. Davies", "T. Mensah", "L. Chen", "P. O'Brien",
    "F. Andersen", "B. Abdullahi"
]

def generate_hs_report():
    incident = random.choice(HS_INCIDENT_TYPES)
    location = random.choice(HS_LOCATIONS)
    person = random.choice(HS_NAMES)
    supervisor = random.choice([n for n in HS_NAMES if n != person])
    severity = random.choice(["Low", "Medium", "High", "Critical"])
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2023, 2024)
    date = f"{day:02d}/{month:02d}/{year}"
    action = random.choice(HS_ACTIONS)

    severity_desc = {
        "Low": "minor",
        "Medium": "moderate",
        "High": "serious",
        "Critical": "critical"
    }[severity]

    text = (
        f"INCIDENT REPORT — Health and Safety\n"
        f"Date of Report: {date}\n"
        f"Location: {location}\n"
        f"A {severity_desc} incident occurred involving {person}. "
        f"The incident was classified as: {incident}. "
        f"The supervising officer {supervisor} was notified immediately. "
        f"The event took place at {location} during normal working hours. "
        f"Action taken: {action}."
    )

    extraction = {
        "incident_type": incident,
        "date_reported": date,
        "location": location,
        "parties_involved": [person, supervisor],
        "severity": severity,
        "summary": f"{incident} involving {person} at {location}.",
        "action_required": action
    }

    return text, extraction


# ── Domain: IT Security ────────────────────────────────────────────────────

IT_INCIDENT_TYPES = [
    "Phishing email — user clicked link",
    "Unauthorised login attempt — account locked",
    "Malware detected on endpoint",
    "Data exfiltration attempt blocked",
    "Ransomware alert — isolated immediately",
    "Privilege escalation detected",
    "Unpatched vulnerability exploited",
    "Brute force attack on VPN gateway",
    "Insider threat — suspicious data access",
    "DDoS attack on public-facing service"
]

IT_SYSTEMS = [
    "Finance ERP system", "Customer database server",
    "VPN gateway", "Email gateway", "Development environment",
    "Cloud storage bucket", "Domain controller",
    "Internal HR portal", "Production API server",
    "Backup storage system"
]

IT_ACTIONS = [
    "Account suspended and password reset enforced",
    "Endpoint quarantined and submitted for forensic analysis",
    "Firewall rules updated to block source IP range",
    "Incident escalated to CISO for review",
    "All-staff phishing awareness notification issued",
    "Vulnerability patched and system restarted",
    "Access logs reviewed for past 30 days",
    "Affected data encrypted and access audited",
    "Third-party security firm engaged for investigation",
    "System restored from verified clean backup"
]

IT_TEAMS = [
    "SOC Analyst Team", "Network Security Team",
    "Endpoint Security Team", "Cloud Security Team",
    "Identity and Access Management Team"
]

def generate_it_report():
    incident = random.choice(IT_INCIDENT_TYPES)
    system = random.choice(IT_SYSTEMS)
    team = random.choice(IT_TEAMS)
    severity = random.choice(["Low", "Medium", "High", "Critical"])
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2023, 2024)
    date = f"{day:02d}/{month:02d}/{year}"
    action = random.choice(IT_ACTIONS)

    text = (
        f"IT SECURITY INCIDENT REPORT\n"
        f"Reported: {date}\n"
        f"Affected System: {system}\n"
        f"Severity: {severity}\n"
        f"Incident type: {incident}. "
        f"The {team} detected and responded to this event. "
        f"The affected system is: {system}. "
        f"Immediate response: {action}. "
        f"All relevant logs have been preserved for investigation."
    )

    extraction = {
        "incident_type": incident,
        "date_reported": date,
        "location": system,
        "parties_involved": [team],
        "severity": severity,
        "summary": f"{incident} on {system}, detected by {team}.",
        "action_required": action
    }

    return text, extraction


# ── Domain: Field Equipment Maintenance ───────────────────────────────────

FM_EQUIPMENT = [
    "HVAC Unit 3B", "Industrial pump assembly",
    "Conveyor belt system", "Emergency generator",
    "Pressure relief valve bank", "Cooling tower unit",
    "Control panel MCC-07", "Compressed air system",
    "Overhead crane assembly", "Fire suppression system"
]

FM_FAULTS = [
    "Bearing failure — excessive vibration detected",
    "Seal degradation — minor fluid leak observed",
    "Electrical fault — circuit breaker tripped",
    "Calibration drift — sensor readings outside tolerance",
    "Corrosion identified on external housing",
    "Scheduled maintenance overdue — performance degraded",
    "Emergency stop triggered — root cause under investigation",
    "Filter blockage — reduced flow rate confirmed",
    "Overheating — thermal protection activated",
    "Component fracture — equipment taken offline"
]

FM_ACTIONS = [
    "Replacement part ordered, ETA 48 hours",
    "Temporary repair applied, permanent fix scheduled",
    "Equipment taken offline pending full inspection",
    "OEM contacted for technical guidance",
    "Maintenance team dispatched for same-day repair",
    "Risk assessment updated before return to service",
    "Monitoring frequency increased to hourly",
    "Third-party inspection booked for next week",
    "Workaround procedure implemented to maintain operations",
    "Full component replacement approved by engineering manager"
]

FM_TECHNICIANS = [
    "Lead Technician R. Morris",
    "Field Engineer A. Nwosu",
    "Senior Technician P. Walsh",
    "Maintenance Lead S. Kumar",
    "Operations Engineer F. Johansson"
]

FM_SITES = [
    "Site Alpha — Manchester", "Site Bravo — Leeds",
    "Site Charlie — Sheffield", "Site Delta — Liverpool",
    "Site Echo — Birmingham"
]

def generate_fm_report():
    equipment = random.choice(FM_EQUIPMENT)
    fault = random.choice(FM_FAULTS)
    technician = random.choice(FM_TECHNICIANS)
    site = random.choice(FM_SITES)
    severity = random.choice(["Low", "Medium", "High", "Critical"])
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2023, 2024)
    date = f"{day:02d}/{month:02d}/{year}"
    action = random.choice(FM_ACTIONS)

    text = (
        f"FIELD MAINTENANCE REPORT\n"
        f"Date: {date}\n"
        f"Site: {site}\n"
        f"Equipment: {equipment}\n"
        f"Fault reported: {fault}. "
        f"Reported by {technician} at {site}. "
        f"Severity assessed as {severity}. "
        f"Recommended action: {action}."
    )

    extraction = {
        "incident_type": fault,
        "date_reported": date,
        "location": f"{equipment} — {site}",
        "parties_involved": [technician],
        "severity": severity,
        "summary": f"{fault} on {equipment} at {site}.",
        "action_required": action
    }

    return text, extraction


# ── Prompt formatting ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document analysis assistant. 
Extract structured information from the incident report below.
Return ONLY a valid JSON object with these fields:
incident_type, date_reported, location, parties_involved, severity, summary, action_required.
Do not include any other text."""

def format_prompt(text, extraction=None):
    """Format as instruction-following prompt for causal LM fine-tuning."""
    prompt = f"{SYSTEM_PROMPT}\n\nReport:\n{text}\n\nJSON:"
    if extraction is not None:
        prompt += f" {json.dumps(extraction)}"
    return prompt


# ── Dataset generation ─────────────────────────────────────────────────────

GENERATORS = {
    "health_safety": generate_hs_report,
    "it_security": generate_it_report,
    "field_maintenance": generate_fm_report
}

def generate_dataset(n_examples, seed=42):
    """Generate a balanced multi-domain extraction dataset."""
    random.seed(seed)
    examples = []
    domains = list(GENERATORS.keys())

    for i in range(n_examples):
        domain = domains[i % len(domains)]
        text, extraction = GENERATORS[domain]()
        examples.append({
            "domain": domain,
            "text": text,
            "extraction": extraction,
            "prompt": format_prompt(text, extraction),
            "input_prompt": format_prompt(text)
        })

    return examples


def save_dataset():
    """Generate and save train/test splits."""
    os.makedirs(DATA_CONFIG["data_dir"], exist_ok=True)

    n_total = DATA_CONFIG["n_train"] + DATA_CONFIG["n_test"]
    all_examples = generate_dataset(n_total, seed=DATA_CONFIG["seed"])

    train = all_examples[:DATA_CONFIG["n_train"]]
    test = all_examples[DATA_CONFIG["n_train"]:]

    train_path = os.path.join(DATA_CONFIG["data_dir"], "train.jsonl")
    test_path = os.path.join(DATA_CONFIG["data_dir"], "test.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    domain_counts = {}
    for ex in train:
        domain_counts[ex["domain"]] = domain_counts.get(ex["domain"], 0) + 1

    print(f"Dataset saved.")
    print(f"  Train: {len(train)} examples — {domain_counts}")
    print(f"  Test:  {len(test)} examples")
    return train, test


if __name__ == "__main__":
    save_dataset()