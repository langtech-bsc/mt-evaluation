import plotly.express as px
from itertools import cycle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd


def is_valid_mt_dataset(name):
    # Check the dataset name against conditions
    return all([
        'hb' not in name,
        name != 'perturbations',
        name != 'must_she',
        'mmhb' not in name
    ])
    
def plot_language_comparison_spider(data, metric, init=0, limit=100, srcxx = 'ca', tgtxx = 'ca'):

    models = data['model_name'].unique()

    # Assign colors to models
    colors = px.colors.qualitative.Dark24  # or any other color set
    color_cycle = cycle(colors)
    model_colors = {model: next(color_cycle) for model in models}

    # Lang as source
    targets_source = data[data['source'] == srcxx]['target'].unique()
    targets_source.sort()

    angles_source = np.linspace(0, 2 * np.pi, len(targets_source), endpoint=False).tolist()
    angles_source += angles_source[:1]  # Complete the loop

    # Lang as target
    sources_target = data[data['target'] == tgtxx]['source'].unique()
    sources_target.sort()

    angles_target = np.linspace(0, 2 * np.pi, len(sources_target), endpoint=False).tolist()
    angles_target += angles_target[:1]  # Complete the loop

    # Create subplots with titles for each polar plot
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}]],
        subplot_titles=(f'{srcxx.upper()}-XX - {metric} Scores by Model', f'XX-{tgtxx.upper()} - {metric} Scores by Model')
    )

    added_models_legend = []
    for model in models:
        # Lang as source
        model_data_source = data[(data['source'] == srcxx) & (data['model_name'] == model)]
        scores_source = [model_data_source[model_data_source['target'] == tgt][metric].values[0] if not model_data_source[model_data_source['target'] == tgt].empty else 0 for tgt in targets_source]
        scores_source += scores_source[:1]  # Complete the loop

        angles_mod_source = [a for i, a in enumerate(angles_source) if scores_source[i] != 0]
        scores_mod_source = [s for i, s in enumerate(scores_source) if s != 0]

        if len(scores_mod_source) > 0: added_models_legend.append(model)

        fig.add_trace(go.Scatterpolar(
            r=scores_mod_source,
            theta=np.degrees(angles_mod_source),
            mode='lines+markers' if len(scores_mod_source) >= 5 else 'markers',
            name=model,
            marker=dict(size=9, color=model_colors[model]),
            line=dict(color=model_colors[model]),
            opacity=0.8,
            legendgroup=model,
            showlegend=True,
        ), row=1, col=1)
        

        # Catalan as target
        model_data_target = data[(data['target'] == tgtxx) & (data['model_name'] == model)]
        scores_target = [model_data_target[model_data_target['source'] == src][metric].values[0] if not model_data_target[model_data_target['source'] == src].empty else 0 for src in sources_target]
        scores_target += scores_target[:1]  # Complete the loop

        angles_mod_target = [a for i, a in enumerate(angles_target) if scores_target[i] != 0]
        scores_mod_target = [s for i, s in enumerate(scores_target) if s != 0]

        fig.add_trace(go.Scatterpolar(
            r=scores_mod_target,
            theta=np.degrees(angles_mod_target),
            mode='lines+markers' if len(scores_mod_target) >= 5 else 'markers',
            name=model,
            marker=dict(size=9, color=model_colors[model]),
            line=dict(color=model_colors[model]),
            opacity=0.8,
            legendgroup=model,
            showlegend=False if model in added_models_legend else True,
        ), row=1, col=2)

    # Update layout for both polar plots
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[init, limit], tickfont=dict(size=10), gridcolor='rgba(0, 0, 0, 0.05)'),
            angularaxis=dict(tickmode='array', tickvals=np.degrees(angles_source[:-1]), ticktext=targets_source)
        ),
        polar2=dict(
            radialaxis=dict(range=[init, limit], tickfont=dict(size=10)),
            angularaxis=dict(tickmode='array', tickvals=np.degrees(angles_target[:-1]), ticktext=sources_target, gridcolor='rgba(0, 0, 0, 0.05)')
        ),
        showlegend=True,
        legend=dict(x=1.1, y=1)
    )

    fig.update_annotations(font_size=12, yshift=30)
    return fig


def plot_segment_pairwise(DATA_MODELS, system1, system2, metric_pairwise):

    system1_scores = DATA_MODELS[system1][metric_pairwise]
    system2_scores = DATA_MODELS[system2][metric_pairwise]

    if len(system1_scores) != len(system2_scores):
        return None

    srcs = DATA_MODELS[system1]['sources']
    refs = DATA_MODELS[system1]['targets']

    translations_s1 = DATA_MODELS[system1]['translations']
    translations_s2 = DATA_MODELS[system2]['translations']

    # Create a DataFrame from the input scores
    df = pd.DataFrame({
        system1: system1_scores,
        system2: system2_scores,
        'Segment index': range(0, len(system1_scores)),
        'Source': srcs,
        'Reference': refs,
        f'Hyp {system1}': translations_s1,
        f'Hyp {system2}': translations_s2,
    })

    # Calculate the absolute differences for bubble sizes
    df['size'] = abs(df[system1] - df[system2])
    # Ensure no zero sizes (which would result in invisible bubbles)
    df['size'] = df['size'].replace(0, 1e-6)

    # Create the bubble plot
    fig = px.scatter(
        df,
        x=system1,
        y=system2,
        size='size',
        size_max=25,  # Adjust as needed for better visualization
        title='Bubble Plot of System Scores',
        hover_data=['Segment index', 'Source', 'Reference', f'Hyp {system1}',  f'Hyp {system2}']
    )

    # Update axis labels
    fig.update_layout(
        xaxis_title=f'{system1} ({metric_pairwise})',
        yaxis_title=f'{system2} ({metric_pairwise})',
        legend_title='Absolute Difference'
    )

    return fig

color_severity = {'critical': 'rgb(239, 83, 80)',
                  'minor': 'rgb(79, 195, 247)',
                  'major': 'rgb(255, 202, 40)'}

def process_sentence(sentence, annotations):
    # Sort annotations by 'start' index
    annotations = sorted(annotations, key=lambda x: x['start'])
    
    result = []
    last_index = 0
    
    for ann in annotations:
        start = ann['start']
        end = ann['end']
        text = ann['text']
        severity = ann['severity']
        
        # Append text before the current annotation
        if last_index < start:
            result.append(sentence[last_index:start])
        
        # Append the annotated text with severity
        result.append((text, severity, color_severity[severity]))
        
        # Update last_index
        last_index = end
    
    # Append any remaining text after the last annotation
    if last_index < len(sentence):
        result.append(sentence[last_index:])
    
    return result


def count_errors(error_spans):

    minor_count, major_count, critical_count = 0, 0, 0

    for item in error_spans:
        for error in item:
            if error['severity'] == 'major':
                major_count += 1
            elif error['severity'] == 'minor':
                minor_count += 1
            elif error['severity'] == 'critical':
                critical_count += 1

    return minor_count, major_count, critical_count


def create_stacked_bar_chart(data_dict):
    # Extracting the categories and values
    models = list(data_dict.keys())
    categories = ['minor', 'major', 'critical']

    # Preparing data for each category
    minor_values = [data_dict[model]['minor'] for model in models]
    major_values = [data_dict[model]['major'] for model in models]
    critical_values = [data_dict[model]['critical'] for model in models]

    # Create the stacked bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Minor',
        x=models,
        y=minor_values,
        marker_color='rgb(79, 195, 247)'
    ))
    fig.add_trace(go.Bar(
        name='Major',
        x=models,
        y=major_values,
        marker_color='rgb(255, 202, 40)'
    ))
    fig.add_trace(go.Bar(
        name='Critical',
        x=models,
        y=critical_values,
        marker_color='rgb(239, 83, 80)'
    ))

    # Update the layout for a stacked bar chart
    fig.update_layout(
        barmode='stack',
        title="Stacked Bar Chart of Model Errors",
        xaxis_title="Models",
        yaxis_title="Number of Errors",
        legend_title="Error Severity"
    )

    return fig


def get_score(segments, index):
    if 0 <= index < len(segments):
        return round(segments[index], 3)
    else:
        return None

def get_string(segments, index):
    if 0 <= index < len(segments):
        return segments[index]
    else:
        return None