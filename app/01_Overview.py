import streamlit as st
from utils import *

DATA_PATH_csv = '../data/results_summary.csv'

def main():
    st.set_page_config(page_title="Overview", page_icon="🍋", layout="wide")

    dataplot = pd.read_csv(DATA_PATH_csv)
    col1chart, col2chart, col3chart = st.columns(3)

    with col1chart:
        dataset_chart_selected = st.selectbox('Select Dataset', dataplot['dataset'].unique())

    with col2chart:
        metric_chart_selected = st.selectbox('Select Metric', ['bleu', 'ter','chrf','comet', 'comet_kiwi','bleurt', 'xcomet', 'metricx', 'metricx_qe'])

    with col3chart:
        selected_model_name = st.multiselect('Select models to compare', options = dataplot.query("dataset == @dataset_chart_selected")['model_name'].unique() )

    with st.sidebar:
        options_src = sorted(dataplot['source'].unique())
        srcxx = st.selectbox('Source language', options=options_src, index=options_src.index('ca') if 'ca' in options_src else 0)
        
        options_tgt = sorted(dataplot['target'].unique())
        tgtxx = st.selectbox('Target language', options=options_tgt, index=options_tgt.index('ca') if 'ca' in options_tgt else 0)

    filtered_dataset = dataplot.query("dataset == @dataset_chart_selected and model_name in @selected_model_name")
    if filtered_dataset.empty:
        st.warning("No data available for the selected dataset and model(s). Please try different options.")
    else:
        selected_limit = filtered_dataset[metric_chart_selected].max() + (5 if metric_chart_selected in ['bleu', 'ter','chrf', 'metricx', 'metricx_qe'] else 0.05)
        fig = plot_language_comparison_spider(filtered_dataset, metric_chart_selected, limit=selected_limit, srcxx = srcxx, tgtxx = tgtxx)
        st.plotly_chart(fig, use_container_width=True, theme=None)



if __name__ == "__main__":
    main()