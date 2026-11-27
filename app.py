"""Historical district-level crop yield demonstration (2013–2017)."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.config import CLEAN_FILE, CROPS, FINAL_TEST_FILE, METADATA_FILE, MODEL_FILE
from src.predict import (
    actual_yield,
    available_states,
    available_years,
    build_model_input,
    district_options,
    load_history,
    load_model,
    predict_yield,
)


@st.cache_data
def read_history():
    return load_history(CLEAN_FILE)


@st.cache_data
def read_model_information():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    metrics = pd.read_csv(FINAL_TEST_FILE)
    required = {"Scope", "MAE_kg_ha", "RMSE_kg_ha", "R2"}
    if not required.issubset(metrics.columns) or not {"Overall", *CROPS}.issubset(metrics["Scope"]):
        raise ValueError("Final evaluation report is incomplete")
    if metadata.get("selected_configuration", {}).get("Model") != "Ridge":
        raise ValueError("The saved model metadata does not identify the expected Ridge model")
    return metadata, metrics.set_index("Scope")


@st.cache_resource
def read_model():
    return load_model(MODEL_FILE)


def show_about_model():
    with st.expander("About This Model"):
        st.write(
            "This Ridge Regression model estimates **district-level** yield for Rice, Wheat, "
            "and Maize. It uses Indian agricultural records from 1978–2017. The app demonstrates "
            "historical one-year-ahead predictions for 2013–2017."
        )
        st.write(
            "Each prediction requires three consecutive years of recorded yields. The model "
            "has not been validated for current-year forecasts or individual farms. Maize had "
            "larger test errors than Rice and Wheat, and unusual past yields can affect later "
            "predictions. Do not treat the result as definitive agricultural or financial advice."
        )


def show_historical_trend(history_table):
    """Plot observed prior yields only; the predicted year is not an actual point."""
    chart_data = history_table.rename(columns={"Historical Yield (kg/ha)": "Yield_kg_ha"})
    base = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Yield_kg_ha:Q", title="Yield (kg/ha)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Year:O", title="Year"),
                     alt.Tooltip("Yield_kg_ha:Q", title="Observed yield (kg/ha)", format=",.2f")],
        )
    )
    chart = (
        base.mark_line(strokeWidth=2, color="#4F8063")
        + base.mark_circle(size=65, color="#4F8063")
    ).properties(height=260)
    st.subheader("Historical Yield Trends")
    st.caption("Observed yields in Y−3 to Y−1. The prediction is not an observed point.")
    st.altair_chart(chart, width="stretch")
    with st.expander("View historical values"):
        st.dataframe(history_table, hide_index=True, width="stretch")


def show_model_performance(test_metrics, crop):
    overall = test_metrics.loc["Overall"]
    crop_metrics = test_metrics.loc[crop]
    st.subheader("Model Performance")
    st.caption("Held-out historical test records, 2013–2017")
    with st.container(border=True):
        mae, rmse, r2 = st.columns(3, gap="medium")
        mae.metric("MAE", f"{overall['MAE_kg_ha']:,.0f} kg/ha")
        rmse.metric("RMSE", f"{overall['RMSE_kg_ha']:,.0f} kg/ha")
        r2.metric("R²", f"{overall['R2']:.3f}")
        st.caption(
            f"For {crop}: MAE {crop_metrics['MAE_kg_ha']:,.0f} kg/ha · "
            f"RMSE {crop_metrics['RMSE_kg_ha']:,.0f} kg/ha"
        )
    st.caption(
        "MAE is the average absolute error; RMSE gives larger mistakes more weight; R² "
        "summarizes fit against a simple average-yield reference. These test metrics are "
        "not guaranteed error bounds for this prediction."
    )


def main():
    st.set_page_config(page_title="Crop Yield Prediction System", layout="centered")
    st.title("Crop Yield Prediction")
    st.caption("Historical crop yield estimates powered by machine learning.")
    st.write(
        "Select a crop, location, and historical year to estimate district-level crop yield "
        "using previous agricultural records."
    )

    try:
        history = read_history()
        metadata, test_metrics = read_model_information()
        model = read_model()
    except FileNotFoundError as error:
        st.error(f"A required project file is missing: {error.filename}. Restore it and restart the app.")
        st.stop()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.error(f"The saved data or model information could not be read: {error}")
        st.stop()
    except Exception:
        st.error("The saved model could not be loaded. Check the model file and installed package versions.")
        st.stop()

    st.divider()
    with st.container(border=True):
        st.subheader("Prediction Parameters")
        crop_column, state_column = st.columns(2, gap="medium")
        crop = crop_column.selectbox("Crop", CROPS)
        states = available_states(history, crop)
        if not states:
            st.info("No states have usable records for this crop.")
            show_about_model()
            return
        state = state_column.selectbox("State", states)
        districts = district_options(history, crop, state)
        if not districts:
            st.info("No districts have usable records for this crop and state.")
            show_about_model()
            return
        district_column, year_column = st.columns(2, gap="medium")
        district_code = district_column.selectbox(
            "District", list(districts),
            format_func=lambda code: f"{districts[code]} (code {code})",
        )
        years = available_years(history, crop, state, district_code)
        if not years:
            st.info("No 2013–2017 prediction year has three consecutive, positive historical yields for this district and crop.")
            show_about_model()
            return
        year = year_column.selectbox("Prediction year", years)
        predict_clicked = st.button("Predict Yield", type="primary")

    if predict_clicked:
        try:
            model_input, history_table = build_model_input(
                history, crop, state, district_code, year, metadata, model
            )
            prediction = predict_yield(model, model_input)
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Prediction is unavailable: {error}")
            return
        except Exception:
            st.error("Prediction failed because the saved model could not process these inputs.")
            return

        st.divider()
        st.subheader("Prediction Results")
        with st.container(border=True):
            predicted_column, previous_column, average_column = st.columns(
                [1.25, 1, 1], gap="medium"
            )
            predicted_column.metric("Predicted Yield", f"{prediction:,.0f} kg/ha")
            previous_column.metric(
                "Previous Year's Yield",
                f"{model_input['Previous_Year_Yield'].iloc[0]:,.0f} kg/ha",
            )
            average_column.metric(
                "Three-Year Average",
                f"{model_input['Previous_3_Year_Avg_Yield'].iloc[0]:,.0f} kg/ha",
            )
        st.caption(f"Estimated district yield for {crop} in {districts[district_code]}, {state} ({year}).")

        show_historical_trend(history_table)

        with st.expander("Historical Evaluation — Actual vs. Predicted"):
            st.caption("Retrospective comparison only. The actual yield was not used to make this prediction.")
            try:
                observed = actual_yield(history, crop, state, district_code, year)
            except ValueError as error:
                st.warning(str(error))
            else:
                if observed is None:
                    st.info("No valid actual yield is recorded for this prediction year.")
                else:
                    st.write(f"Recorded actual yield: **{observed:,.0f} kg/ha**")
                    st.write(f"Predicted yield: **{prediction:,.0f} kg/ha**")
                    st.write(f"Absolute prediction error: **{abs(prediction - observed):,.0f} kg/ha**")
                    st.caption("Absolute error is the size of the difference between this historical prediction and the recorded yield.")

        show_model_performance(test_metrics, crop)

    show_about_model()


if __name__ == "__main__":
    main()
