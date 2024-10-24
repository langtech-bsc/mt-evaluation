import streamlit as st
import pandas as pd

DATA_PATH_csv = '../results_summary/results_summary.csv'
DATA_PATH = '../results'


def main():

    st.set_page_config(page_title="Gender Bias", page_icon="🍋", layout="wide")

    # Tabs
    tab1, tab2 = st.tabs([
        "Must-she", 
        "MMHB"
    ])

    df = pd.read_csv(DATA_PATH_csv)

    with tab1:
        
        df_must_she = df.loc[df['dataset'] == 'must_she']

        target_col, models_selected = st.columns(2)

        with target_col:
            target_selected = st.selectbox('Select Target Language', sorted( df_must_she['target'].unique() ) )

        df_must_she_filtered = df_must_she.loc[df_must_she['target'] == target_selected]

        with models_selected:
            selected_models = st.multiselect('Select models to compare', options = df_must_she_filtered['model_name'].unique())


    with tab2:
        pass

if __name__ == "__main__":
    main()