
import streamlit as st
import pandas as pd
import joblib
from pydantic import BaseModel, Field, ValidationError


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Room Type Predictor",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("Model_Pipeline.pkl")


model = load_model()


# --------------------------------------------------
# Feature Columns
# --------------------------------------------------

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood"
]


# --------------------------------------------------
# Pydantic Validation Model
# --------------------------------------------------

class Features(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude coordinate"
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude coordinate"
    )

    price: float = Field(
        ...,
        gt=0,
        description="Price per night, must be positive"
    )

    minimum_nights: int = Field(
        ...,
        ge=1,
        le=365,
        description="Minimum nights required for booking"
    )

    number_of_reviews: int = Field(
        ...,
        ge=0,
        description="Total number of reviews"
    )

    reviews_per_month: float = Field(
        ...,
        ge=0,
        description="Average reviews per month"
    )

    calculated_host_listings_count: int = Field(
        ...,
        ge=0,
        description="Number of listings by this host"
    )

    availability_365: int = Field(
        ...,
        ge=0,
        le=365,
        description="Days available out of 365"
    )

    neighbourhood_group: str = Field(
        ...,
        min_length=1,
        description="Borough or neighbourhood group"
    )

    neighbourhood: str = Field(
        ...,
        min_length=1,
        description="Specific neighbourhood name"
    )


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 Airbnb Room Type Predictor")

st.write(
    "Enter the property details below to predict the room type."
)


# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Property Information")

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060
        )

        price = st.number_input(
            "Price per Night",
            min_value=0.01,
            value=100.0
        )

        minimum_nights = st.number_input(
            "Minimum Nights",
            min_value=1,
            max_value=365,
            value=1,
            step=1
        )

        number_of_reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            value=10,
            step=1
        )

    with col2:

        reviews_per_month = st.number_input(
            "Reviews per Month",
            min_value=0.0,
            value=1.0
        )

        calculated_host_listings_count = st.number_input(
            "Host Listings Count",
            min_value=0,
            value=1,
            step=1
        )

        availability_365 = st.number_input(
            "Availability (365 Days)",
            min_value=0,
            max_value=365,
            value=100,
            step=1
        )

        neighbourhood_group = st.text_input(
            "Neighbourhood Group",
            placeholder="Example: Manhattan"
        )

        neighbourhood = st.text_input(
            "Neighbourhood",
            placeholder="Example: Harlem"
        )

    submitted = st.form_submit_button(
        "🔮 Predict Room Type",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    try:

        # ------------------------------------------
        # Validate Input
        # ------------------------------------------

        features = Features(
            latitude=latitude,
            longitude=longitude,
            price=price,
            minimum_nights=minimum_nights,
            number_of_reviews=number_of_reviews,
            reviews_per_month=reviews_per_month,
            calculated_host_listings_count=calculated_host_listings_count,
            availability_365=availability_365,
            neighbourhood_group=neighbourhood_group,
            neighbourhood=neighbourhood
        )

        # ------------------------------------------
        # Convert to DataFrame
        # ------------------------------------------

        row = pd.DataFrame(
            [features.model_dump()],
            columns=COLUMNS
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(row)

        probability = model.predict_proba(row)

        predicted_room = prediction[0]

        probabilities = probability[0]

        # ------------------------------------------
        # Display Result
        # ------------------------------------------

        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Room Type: {predicted_room}"
        )

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Room Type": model.classes_,
            "Probability": probabilities
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

    except ValidationError as e:

        st.error("Please correct the following input errors:")

        for error in e.errors():

            field = error["loc"][0]
            message = error["msg"]

            st.warning(
                f"{field}: {message}"
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )

