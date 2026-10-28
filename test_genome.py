
from genome import build_genome, describe_relationship, get_genome_summary

# Fictional test cases — not real scam reports.
cases = [
    {
        "id": "CASE-001",
        "text": "Your parcel is delayed. Pay a delivery fee using this link.",
        "domains": ["fake-delivery.example"],
        "phone_numbers": ["9000000001"],
        "payment_ids": [],
        "timestamp": "2026-10-20",
        "label": "synthetic_test",
    },
    {
        "id": "CASE-002",
        "text": "Delivery failed. Complete the fee payment to receive your parcel.",
        "domains": ["fake-delivery.example"],
        "phone_numbers": ["9000000001"],
        "payment_ids": [],
        "timestamp": "2026-10-21",
        "label": "synthetic_test",
    },
    {
        "id": "CASE-003",
        "text": "Your electricity bill is ready. View your monthly usage.",
        "domains": ["electricity-info.example"],
        "phone_numbers": ["9000000002"],
        "payment_ids": [],
        "timestamp": "2026-10-22",
        "label": "synthetic_test",
    },
]

graph = build_genome(cases)

print("\n--- GENOME SUMMARY ---")
print(get_genome_summary(graph))

print("\n--- CASE 001 vs CASE 002 ---")
print(describe_relationship(graph, "CASE-001", "CASE-002"))

print("\n--- CASE 001 vs CASE 003 ---")
print(describe_relationship(graph, "CASE-001", "CASE-003"))
