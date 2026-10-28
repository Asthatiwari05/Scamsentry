
from mutation import compare_messages, detect_mutation_candidates

messages = {
    "VERSION-A": "Your parcel is delayed. Pay a delivery fee.",
    "VERSION-B": "Delivery failed. Submit the required charge to receive your package.",
    "UNRELATED": "Your electricity bill is ready. View your monthly usage.",
}

print("\n--- ALL MESSAGE COMPARISONS ---")
for result in compare_messages(messages):
    print(result)

print("\n--- MUTATION CANDIDATES ---")
for result in detect_mutation_candidates(messages, threshold=0.10):
    print(result)
