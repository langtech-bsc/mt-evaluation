import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH_csv = '../results_summary/results_summary.csv'
DATA_PATH = '../results'


def main():

    st.set_page_config(page_title="Perturbations", page_icon="🔎", layout="wide")
    
    if os.path.exists(DATA_PATH_csv):
        # Load the dataset
        df = pd.read_csv(DATA_PATH_csv)
        df = df.loc[df['dataset'] == 'perturbations']

        col1, col2 = st.columns(2)

        with col1:
            source_selected = st.selectbox('Select Source Language', sorted( df['source'].unique() ) )

        # Filter the DataFrame based on the selected dataset and source
        filtered_df_by_source = df[df['source'] == source_selected]

        with col2:
            target_selected = st.selectbox('Select Target Language',  sorted( filtered_df_by_source['target'].unique() ) )

        # Filter dataset by target
        filtered_df_by_target = filtered_df_by_source[filtered_df_by_source['target'] == target_selected]
        
        # Display the filtered rows
        st.write('Filtered Rows:')
        st.dataframe(filtered_df_by_target)


        JSONS = []
        NAMES = []
        if not filtered_df_by_target.empty:    
            for index, row in filtered_df_by_target.iterrows():
                # Get the model_name and file_name
                model_name = row['model_name']
                file_name = row['file_name']
                
                file_path = os.path.join(DATA_PATH, model_name, file_name)
                JSONS.append(file_path)
                NAMES.append(model_name)
        else:
            st.write('No data available for the selected combination.')

        # Load the JSON data
        def load_data(json_path):
            with open(json_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            first_key = next(iter(data['results']))
            return data['results'][first_key]

        col1, col2 = st.columns(2)

        with col1:
            selected_models = st.multiselect('Select models to compare', options = NAMES)

        with col2:
            metrics_selected = st.selectbox("Select Metric for Visualization", ["bleu", "ter", "comet"])

        # load models data
        DATA_MODELS = {}

        for modelname, json_path in zip(NAMES, JSONS):
            if modelname in selected_models:
                data = load_data(json_path)

                bleu_corpus = data.get('bleu,none', None)
                ter_corpus = data.get('ter,none', None)
                comet_corpus = data.get('comet,none', None)

                DATA_MODELS[modelname] = {  
                                            # metrics
                                            'bleu': bleu_corpus,
                                            'ter': ter_corpus,
                                            'comet': comet_corpus
                                        }

        if DATA_MODELS:
            
            fig = make_subplots(
                                cols=3, 
                                rows=1, 
                                subplot_titles=["Swap", "CharDupe", "CharDrop"],
                                shared_xaxes=True,
                                shared_yaxes=True
                            )

            plot_types = ["swap", "chardupe", "chardrop"]
            for col, plot_type in enumerate(plot_types, start=1):
                for model_name, metrics in DATA_MODELS.items():
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 0.25, 0.5, 0.75, 1],
                            y=metrics[metrics_selected][plot_type],
                            mode="lines+markers",
                            name=model_name,
                            legendgroup=model_name
                        ),
                        row=1,
                        col=col
                    )
                
            # Layout adjustments
            fig.update_layout(
                title=f"{plot_type.capitalize()} ({metrics_selected.upper()})",
                xaxis_title="Noise",
                yaxis_title=f"{metrics_selected.upper()} Score",
                legend_title="Model",
                template="plotly_white"
            )
            
            # Display the plot
            st.plotly_chart(fig)

        else:
            st.warning("No data available for visualizations.")


    else:
        st.error(f"File not found: {DATA_PATH_csv}. Please ensure the file exists.")

if __name__ == "__main__":
    main()