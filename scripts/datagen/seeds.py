"""Seed template bank used as a fallback and as few-shot examples for the LLM.

Templates use [LABEL] placeholders drawn from `labels.ALLOWED_LABELS`. They span
several domains so a model trained purely on seeds still sees variety; the LLM
step expands this substantially.
"""

SEED_TEMPLATES = [
    # Clinical
    "Patient [GIVEN_NAME] [SURNAME] (DOB [DATE_OF_BIRTH], MRN [MEDICAL_RECORD_NUMBER]) was admitted to [CITY] General on [DATE].",
    "Dr. [SURNAME] reviewed the labs for [GIVEN_NAME] [SURNAME]; follow-up scheduled [DATE] at [TIME].",
    "Contact the patient at [PHONE] or [EMAIL]; insurance ID [GOVERNMENT_ID].",
    "[GIVEN_NAME] [SURNAME], age [AGE], [GENDER], presented with chest pain at [TIME].",
    # Finance
    "Please wire the payment to account [ACCOUNT_NUMBER], routing [ROUTING_NUMBER], IBAN [IBAN].",
    "Card [CREDIT_DEBIT_CARD] (CVV [CVV]) was charged; contact [GIVEN_NAME] [SURNAME] at [EMAIL].",
    "Customer [CUSTOMER_ID] disputed a transaction; SWIFT [SWIFT_BIC], tax id [TAX_ID].",
    "Your PIN is [PIN]. Never share it. Questions? Call [PHONE].",
    # Contact / forms
    "Name: [GIVEN_NAME] [SURNAME]\nAddress: [BUILDING_NUMBER] [STREET_NAME], [CITY], [STATE] [ZIP_CODE]\nPhone: [PHONE]",
    "Ship to [GIVEN_NAME] [SURNAME], [STREET_ADDRESS], [SECONDARY_ADDRESS], [CITY] [ZIP_CODE], [COUNTRY].",
    "Register [EMAIL] with username [USERNAME]; temporary password [PASSWORD].",
    "Employee [GIVEN_NAME] [SURNAME] ([EMPLOYEE_ID]) starts [DATE]; badge mailed to [STREET_ADDRESS].",
    # Chat / email prose
    "Hey, it's [GIVEN_NAME] — my new number is [PHONE] and email [EMAIL]. Talk soon!",
    "Forwarding [GIVEN_NAME] [SURNAME]'s details: SSN [SSN], DOB [DATE_OF_BIRTH], license [DRIVERS_LICENSE].",
    "The delivery driver ([GIVEN_NAME]) will arrive around [TIME] at [BUILDING_NUMBER] [STREET_NAME].",
    # Tech / logs
    "User [USERNAME] logged in from [IPV4] (device MAC [MAC_ADDRESS]) at [TIME].",
    "API request from [IPV6] used key [API_KEY]; see dashboard at [URL].",
    "Server [URL] flagged account [EMAIL] after 3 failed logins from [IPV4].",
    # Legal / travel / vehicle
    "Passport [PASSPORT] issued to [GIVEN_NAME] [SURNAME], [COUNTRY], expires [DATE].",
    "Vehicle plate [LICENSE_PLATE] registered to [GIVEN_NAME] [SURNAME], [STREET_ADDRESS].",
    "[COMPANY_NAME] invoice for [GIVEN_NAME] [SURNAME], due [DATE], remit to [IBAN].",
]

# Templates with no PII — hard negatives to reduce false positives.
NEGATIVE_TEMPLATES = [
    "The quarterly report shows a 12% increase in regional sales volume.",
    "Please restart the service and confirm the health check passes.",
    "Order #48213 shipped and is expected to arrive within five business days.",
    "The meeting agenda covers roadmap, hiring, and the Q3 budget review.",
    "Turn left after the second traffic light and continue for two miles.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Model accuracy improved after tuning the learning rate and batch size.",
    "Ticket 9921 was closed as a duplicate of the networking incident.",
]
