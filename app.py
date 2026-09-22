import streamlit as st
import requests
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NYC Room Type Predictor",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# FASTAPI URL
# =========================================================

API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# HEADER
# =========================================================

st.title("🏙️ NYC Room Type Predictor")

st.write(
    "Enter Airbnb property details below to predict "
    "the room type using a machine learning model."
)

st.divider()


# =========================================================
# INPUT SECTION — TOP
# =========================================================

st.header("🏠 Property Information")

st.write(
    "Provide the property details required by the prediction model."
)


# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------

st.subheader("📍 Location")

col1, col2, col3, col4 = st.columns(4)

with col1:
    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ]
    )

with col2:
    neighbourhood = st.text_input(
        "Neighbourhood",
        value="Midtown"
    )

with col3:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.7128,
        format="%.6f"
    )

with col4:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-74.0060,
        format="%.6f"
    )


# ---------------------------------------------------------
# PRICING & BOOKING
# ---------------------------------------------------------

st.subheader("💰 Pricing & Booking")

col1, col2, col3 = st.columns(3)

with col1:
    price = st.number_input(
        "Price per Night ($)",
        min_value=0.01,
        value=150.0,
        step=10.0
    )

with col2:
    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=365,
        value=2
    )

with col3:
    availability_365 = st.number_input(
        "Availability (Days / Year)",
        min_value=0,
        max_value=365,
        value=200
    )


# ---------------------------------------------------------
# REVIEWS & HOST
# ---------------------------------------------------------

st.subheader("⭐ Reviews & Host")

col1, col2, col3 = st.columns(3)

with col1:
    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=25
    )

with col2:
    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        value=1.5,
        step=0.1
    )

with col3:
    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=0,
        value=1
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Room Type",
    use_container_width=True
)


# =========================================================
# PREDICTION SECTION — BOTTOM
# =========================================================

st.divider()

st.header("🎯 Prediction Result")


# =========================================================
# INITIAL STATE
# =========================================================

if not predict_button:

    st.info(
        "Enter the property information above and click "
        "**Predict Room Type** to generate a prediction."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "ML Classifier"
        )

    with col2:
        st.metric(
            "Target",
            "Room Type"
        )

    with col3:
        st.metric(
            "Backend",
            "FastAPI"
        )


# =========================================================
# RUN PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE API PAYLOAD
    # -----------------------------------------------------

    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count":
            calculated_host_listings_count,
        "availability_365": availability_365,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood
    }


    # -----------------------------------------------------
    # CALL FASTAPI
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Running machine learning prediction..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )


        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            result = response.json()

            prediction = result["Predicted_room_type"]

            probabilities = result["Probability"]

            classes = result.get(
                "Classes",
                [
                    "Entire home/apt",
                    "Private room",
                    "Shared room"
                ]
            )

            confidence = max(probabilities) * 100


            # =================================================
            # MAIN RESULT
            # =================================================

            st.success(
                "Prediction generated successfully!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Room Type",
                    prediction
                )

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "Prediction Status",
                    "Successful"
                )


            # =================================================
            # PROBABILITY SECTION
            # =================================================

            st.divider()

            st.subheader("📊 Prediction Probability")

            probability_df = pd.DataFrame(
                {
                    "Room Type": classes,
                    "Probability": probabilities
                }
            )

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            )


            col1, col2 = st.columns(2)


            # -------------------------------------------------
            # CHART
            # -------------------------------------------------

            with col1:

                chart_df = probability_df[
                    ["Room Type", "Probability (%)"]
                ].set_index("Room Type")

                st.bar_chart(chart_df)


            # -------------------------------------------------
            # TABLE
            # -------------------------------------------------

            with col2:

                display_df = probability_df[
                    ["Room Type", "Probability (%)"]
                ].copy()

                display_df["Probability (%)"] = (
                    display_df["Probability (%)"]
                    .round(2)
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # INPUT SUMMARY
            # =================================================

            st.divider()

            st.subheader("📋 Property Summary")

            col1, col2 = st.columns(2)


            # -------------------------------------------------
            # LOCATION SUMMARY
            # -------------------------------------------------

            with col1:

                st.write("### 📍 Location")

                st.write(
                    f"**Neighbourhood Group:** "
                    f"{neighbourhood_group}"
                )

                st.write(
                    f"**Neighbourhood:** "
                    f"{neighbourhood}"
                )

                st.write(
                    f"**Latitude:** "
                    f"{latitude:.6f}"
                )

                st.write(
                    f"**Longitude:** "
                    f"{longitude:.6f}"
                )


            # -------------------------------------------------
            # PROPERTY SUMMARY
            # -------------------------------------------------

            with col2:

                st.write("### 🏠 Property")

                st.write(
                    f"**Price:** ${price:.2f} / night"
                )

                st.write(
                    f"**Minimum Nights:** "
                    f"{minimum_nights}"
                )

                st.write(
                    f"**Number of Reviews:** "
                    f"{number_of_reviews}"
                )

                st.write(
                    f"**Reviews per Month:** "
                    f"{reviews_per_month:.2f}"
                )

                st.write(
                    f"**Availability:** "
                    f"{availability_365} days"
                )

                st.write(
                    f"**Host Listings:** "
                    f"{calculated_host_listings_count}"
                )


        # =================================================
        # API ERROR
        # =================================================

        else:

            st.error(
                f"FastAPI returned error "
                f"{response.status_code}"
            )

            st.code(response.text)


    # =====================================================
    # CONNECTION ERROR
    # =====================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.write(
            "Start the FastAPI backend first:"
        )

        st.code(
            "python -m uvicorn main:app --reload"
        )


    # =====================================================
    # TIMEOUT
    # =====================================================

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The request timed out."
        )


    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NYC Room Type Prediction System | "
    "Machine Learning + FastAPI + Streamlit"
)