import pickle


# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Example new customer
# Format: [industry_code, role_code, interest_code]
new_customer = [[1, 1, 1]]


# Predict
result = model.predict(new_customer)


print("Prediction Result:", result[0])


# Generate Email
if result[0] == 1:

    email = """
Subject: Solution for Your Business Growth

Dear Customer,

We noticed your interest in our services.
Our team has solutions designed for your business needs.

We would love to connect with you and discuss more.

Best Regards,
AI Sales Team
"""

    print("\nSEND EMAIL ✅")
    print(email)

else:
    print("\nSKIP CUSTOMER ❌")
