import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import bigrams, trigrams
from collections import Counter
import nltk
import re
import plotly.express as px
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('datasets/Characters.csv', delimiter=';')

def clean_blood_status(value):
    if isinstance(value, str):
        normalized_value = re.sub(r'\s+', ' ', value.lower().strip())
        if 'pure-blood or half-blood' in normalized_value:
            return 'Pure-blood or half-blood'
        elif 'half-blood or pure-blood' in normalized_value:
            return 'Pure-blood or half-blood'
        elif 'pure-blood or half-blood' in normalized_value:
            return 'Pure-blood or half-blood'
        elif 'pure-blood or half-blood' in normalized_value:
            return 'Pure-blood or half-blood'
        elif 'muggle-born or half-blood[' in normalized_value:
            return 'Muggle-born or half-blood'
        elif 'half-blood[' in normalized_value:
            return 'Half-blood'
        return value
    return value

df['Blood status'] = df['Blood status'].apply(clean_blood_status)
df['Species'] = df.Species.str.strip()

def clean_loyalty(value):
    if isinstance(value, str):
        normalized_value = re.sub(r'\s+', ' ', value.strip())  # Normalize by removing extra spaces
        if 'lord voldemort' in normalized_value.lower() and 'death eaters' in normalized_value.lower():
            return 'Lord Voldemort| Death Eaters'
        return value
    return value

df['Loyalty'] = df['Loyalty'].apply(clean_loyalty)

# Combined replace
df['Loyalty'] = df['Loyalty'].replace({
    'Order of the Phoenix| British Ministry of Magic': 'Ministry of Magic | Order of the Phoenix',
    'Lord Voldemort': 'Lord Voldemort| Death Eaters'
}, regex=True)

idf = df.copy()

def preprocess_loyalty(loyalty_str):
    if pd.isna(loyalty_str):
        return loyalty_str
    elif "Dumbledore's Army" in loyalty_str and "Hogwarts" in loyalty_str and "Order of the Phoenix" in loyalty_str:
        return "Dumbledore's Army + Order of Phoenix + Hogwarts"
    elif "Dumbledore's Army" in loyalty_str and "Hogwarts" in loyalty_str:
        return "Dumbledore's Army + Hogwarts"
    elif "Lord Voldemort" in loyalty_str and "Death Eaters" in loyalty_str:
        return "Lord Voldemort"
    elif "Hogwarts School of Witchcraft and Wizardry" in loyalty_str:
        return "Hogwarts"
    else:
        return loyalty_str

idf['Loyalty'] = idf['Loyalty'].apply(preprocess_loyalty)

# Create interactive dashboard function
def create_dashboard(house_value, loyalty_value):
    filtered_data = (
        idf[
            (idf.House.isin(house_value)) &
            (idf.Loyalty.isin(loyalty_value))
        ]
        .groupby(['House'])['Loyalty'].count()
        .reset_index()
    )
    # Create a Plotly figure using Plotly Express
    fig = px.bar(filtered_data, x='House', y='Loyalty', title=f'Distribution for House and {loyalty_value} Loyalty')

    return fig  # Return the Plotly figure

# Available options for house and loyalty
available_houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
available_loyalties = ['Order of the Phoenix', 'Lord Voldemort', "Dumbledore's Army + Hogwarts",
                        "Dumbledore's Army + Order of Phoenix + Hogwarts", "Hogwarts", "Original Order of the Phoenix",
                        "Dumbledore's Army", "Minister of Magic"]

## Harry Potter Movie 1

df1 = pd.read_csv('datasets/Harry Potter 1.csv', delimiter=";")
df1['Character'] = df1['Character'].str.strip()
df1['Character'] = df1['Character'].replace({
    'OIiver': 'Oliver',
    'Hermoine': 'Hermione',
    'Draco': 'Malfoy',
    'Lee  Jordan': 'Lee Jordan',

}, regex=True)
stop_words = set(stopwords.words("english"))


def clean(sentence):
    sentence = sentence.lower()  # converting to lower case
    sentence = re.sub("@[A-Za-z0-9_]+", "", sentence)  # removing mentions
    sentence = re.sub("#[A-Za-z0-9_]+", "", sentence)  # removing hashtags
    sentence = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", sentence)  # removing links
    sentence = re.sub('[()!?]', '', sentence)  # removing punctuation
    sentence = re.sub('\[.*?\]', '', sentence)  # removing punctuation
    sentence = re.sub("[^a-z0-9]", " ", sentence)  # removing non-alphanumeric characters
    sentence = sentence.split()  # simple word tokenization
    sentence = [w for w in sentence if not w in stop_words]  # removing stopwords
    sentence = " ".join(word for word in sentence)
    return sentence

df1['clean_sentence'] = df1['Sentence'].map(lambda x: clean(x))

## Harry Potter Movie 2
df2 = pd.read_csv('datasets/Harry Potter 2.csv', delimiter=";")
df2['Character'] = df2.Character.str.strip()
df2['Character'] = df2['Character'].apply(lambda x: x.title())
df2.loc[df2['Character'].str.contains('Lockhart'), 'Character'] = 'Gilderoy Lockhart'
df2['Character'] = df2['Character'].replace({
    'Mcgonagall': 'McGonagall',
    'Uncle Vernon': 'Vernon',
    'Aunt Petunia': 'Petunia',
    'Harry And Ron': 'Ron and Harry'

}, regex=True)

df2['clean_sentence'] = df2['Sentence'].map(lambda x: clean(x))

## Harry Potter Movie 3
df3 = pd.read_csv('datasets/Harry Potter 3.csv', delimiter=";")
df3 = df3.rename(columns={'CHARACTER': 'Character', 'SENTENCE': 'Sentence'})
df3['Character'] = df3.Character.str.strip()
df3['Character'] = df3['Character'].apply(lambda x: x.title())
df3['Character'] = df3['Character'].replace({
    'Mcgonagall': 'McGonagall',
    'Uncle Vernon': 'Vernon',
    'Aunt Petunia': 'Petunia',

}, regex=True)
df3['clean_sentence'] = df3['Sentence'].map(lambda x: clean(x))

df1['Part'] = "Philosopher's Stone"
df2['Part'] = "Chamber of Secrets"
df3['Part'] = "Prisoner of Azkaban"

lst = [df1, df2, df3]  # List of your dataframes
df4 = pd.concat(lst, ignore_index=True)

# Sidebar title
st.sidebar.title("Harry Potter Analysis")

# Sidebar widgets for selecting the movie
movie_choice = st.sidebar.selectbox("Select Movie",
                                    ["Overall Information", "Harry Potter and the Philosopher's Stone",
                                     "Harry Potter and the Chamber of Secrets", "Harry Potter and the Prisoner of Azkaban"])  # Add more movies as needed

with st.sidebar:
    if movie_choice == "Overall Information":
        analysis_choice = st.selectbox("Select Analysis", ["General Distributions", "Interactive Dashboard",
                                                           "Combined Movies Character Analysis"])
    else:
        analysis_choice = st.selectbox("Select Analysis",
                                       ["Word Cloud", "Word Frequency", "Bigrams", "Trigrams", "Character Analysis"])

# Load the corresponding dataset based on the selected movie
if movie_choice == "Harry Potter and the Philosopher's Stone":
    current_df = df1  # Load your dataset for Movie 1
elif movie_choice == "Harry Potter and the Chamber of Secrets":
    current_df = df2  # Load your dataset for Movie 2
elif movie_choice == "Harry Potter and the Prisoner of Azkaban":
    current_df = df3  # Load your dataset for Movie 3

# Main content area
st.title(f"{movie_choice} Analysis")

# Display analysis based on user choice
if movie_choice == "Overall Information" and analysis_choice == "General Distributions":
    st.header("General Distributions")

    # Gender Donut Chart
    gender_counts = df['Gender'].value_counts()
    st.write("Gender Distribution")
    fig_gender = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=0.4,
                                        textinfo='percent+label', marker=dict(colors=['#00008B', '#AA336A']))])
    st.plotly_chart(fig_gender)

    # House Donut Chart
    house_counts = df['House'].value_counts()
    st.write("House Distribution")
    fig_house = go.Figure(data=[go.Pie(labels=house_counts.index, values=house_counts.values, hole=0.4,
                                        textinfo='percent+label', marker=dict(
            colors=['#740001', '#1A472A', '#0E1A40', '#FFD800', '#946B2D', '#000000']))])
    st.plotly_chart(fig_house)

    # Species Bar Plot
    species_counts = df['Species'].value_counts()
    species_counts = species_counts.sort_values(ascending=True)
    st.write("Species Distribution")
    fig_species = go.Figure(data=[go.Bar(x=species_counts.values, y=species_counts.index, text=species_counts.values,
                                          orientation='h', marker=dict(color='#0067a5'))])
    st.plotly_chart(fig_species)

    # Blood Status Bar Plot
    blood_status_counts = df['Blood status'].value_counts()
    blood_status_counts = blood_status_counts.sort_values(ascending=True)
    st.write("Blood Status Distribution")
    fig_blood = go.Figure(data=[go.Bar(x=blood_status_counts.values, y=blood_status_counts.index,
                                       text=blood_status_counts.values, orientation='h', marker=dict(color='#b31b1b'))])
    st.plotly_chart(fig_blood)


elif movie_choice == "Overall Information" and analysis_choice == "Interactive Dashboard":
    st.header("Harry Potter Interactive Dashboard")
    st.write("Use the filters below to explore the Interactive Dashboard.")

    # Sidebar widgets for the Interactive Dashboard
    house_key = "house_filter"  # Unique key for the house filter
    loyalty_key = "loyalty_filter"  # Unique key for the loyalty filter

    house = st.multiselect(
        "Select House",
        available_houses,
        default=available_houses,  # Set default to all available houses
        key=house_key  # Assign a unique key
    )

    loyalty = st.multiselect(
        "Select Loyalty",
        available_loyalties,
        default=available_loyalties,  # Set default to all available loyalties
        key=loyalty_key  # Assign a unique key
    )
    st.plotly_chart(create_dashboard(house, loyalty))

elif movie_choice == "Overall Information" and analysis_choice == "Combined Movies Character Analysis":
    # Create a DataFrame with character dialogue counts
    st.header("Combined First 3 Movies Character Analysis")

    df5 = df4.groupby(['Character', 'Part']).size().reset_index(name='Freq')

    # Filter characters
    selected_characters = ["Harry", "Ron", "Hermione", "Hagrid", "Dumbledore", "Lupin", "McGonagall",
                           "Draco", "Gilderoy Lockhart", "Snape", "Lucius Malfoy",
                           "Mrs. Weasley", "Tom Riddle", "Sirius", "Dobby"]

    characters_filtered = df5[df5['Character'].isin(selected_characters)]

    # Arrange characters by frequency
    character_sorted = characters_filtered.sort_values(by='Freq', ascending=False)

    # Define color map
    color_map = {
        "Philosopher's Stone": "darkred",
        "Chamber of Secrets": "goldenrod",
        "Prisoner of Azkaban": "darkgreen"
    }

    # Create a new column for the total sum of all parts for each character
    character_sorted['Total'] = character_sorted.groupby('Character')['Freq'].transform('sum')

 
    # Plotting using Plotly
    fig = px.bar(character_sorted,
                x='Freq',
                y='Character',
                color='Part',
                color_discrete_map=color_map,
                orientation='h',
                width = 800,
                height = 800,
                hover_data={'Part': True, 'Freq': True},
                labels={'Total': 'Total Sentences'},
                title='Characters with the Most Sentences',
                category_orders={"Character": selected_characters})

    # Customize layout for stacked bar chart
    fig.update_layout(barmode='stack', legend_title_text='Part of Movie Series')
    fig.update_traces(textposition='outside')  # Place text outside the bars

    # Adjust text position for the total
    for char in selected_characters:
        char_data = character_sorted[character_sorted['Character'] == char]
        fig.add_annotation(text=str(char_data['Total'].values[0]),
                        x=char_data['Total'].values[0] + 5,
                        y=char,
                        xanchor='left',
                        showarrow=False)
    st.plotly_chart(fig)

else:
    if analysis_choice == "Word Cloud":
        st.header("Word Cloud")
        # Combine all cleaned sentences into a single string
        all_sentences = ' '.join(current_df['clean_sentence'])
        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=150).generate(all_sentences)
        # Plot the WordCloud image
        st.image(wordcloud.to_array(), use_column_width=True)

    elif analysis_choice == "Word Frequency":
        st.header("Word Frequency")
        # Tokenize the text
        tokens = word_tokenize(" ".join(current_df['clean_sentence']))
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
        # Calculate word frequencies
        word_freq = Counter(tokens)
        # Create a DataFrame of the most common words
        common_words = pd.DataFrame(word_freq.most_common(50), columns=['words', 'count'])
        # Plot horizontal bar graph
        common_words_sorted = common_words.sort_values(by='count', ascending=True)
        # Create a bar chart using Plotly Express
        fig = px.bar(common_words_sorted, x='count', y='words', orientation='h', text='count', color = 'count', color_continuous_scale='Inferno')

        # Customize the chart layout
        fig.update_layout(
            title='Top 50 Word Frequency (Most to Least)',
            xaxis_title='Count',
            yaxis_title='Words',
            showlegend=False,
            autosize = True,
            width = 750,
            height = 900  # Hide the legend
        )

        # Place text outside the bars
        fig.update_traces(textposition='outside')

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

    elif analysis_choice == "Bigrams":
        st.header("Bigrams")
        # Tokenize the text
        tokens = word_tokenize(" ".join(current_df['clean_sentence']))
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
        # Calculate bigram frequencies
        bi_grams = list(bigrams(tokens))
        bigram_freq = Counter(bi_grams)
        # Create a DataFrame of the most common bigrams
        common_bigrams = pd.DataFrame(bigram_freq.most_common(40), columns=['bigram', 'count'])
        common_bigrams_sorted = common_bigrams.sort_values(by='count', ascending=True)
        # Convert the 'Bigram' column to strings
        common_bigrams_sorted['bigram'] = common_bigrams_sorted['bigram'].apply(str)

        fig = px.bar(common_bigrams_sorted, x='count', y='bigram', orientation='h', color='count', text = 'count',color_continuous_scale='Inferno')

        # Update the tickvals and ticktext to match the reversed order
        tickvals = list(range(len(common_bigrams_sorted['bigram'])))
        ticktext = common_bigrams_sorted['bigram']

        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)

        fig.update_layout(
            title='Top 40 Bigrams Frequency (Most to Least)',
            xaxis_title='Count',
            yaxis_title='Bigram',
            showlegend=False,
            autosize=True,
            width=800,
            height=800,
        )

        # Place text outside the bars
        fig.update_traces(textposition='outside')

        st.plotly_chart(fig)

    elif analysis_choice == "Trigrams":
        st.header("Trigrams")
        # Tokenize the text
        tokens = word_tokenize(" ".join(current_df['clean_sentence']))
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
        # Calculate bigram frequencies
        tri_grams = list(trigrams(tokens))
        trigram_freq = Counter(tri_grams)
        # Create a DataFrame of the most common bigrams
        common_trigrams = pd.DataFrame(trigram_freq.most_common(40), columns=['trigram', 'count'])
        common_trigrams_sorted = common_trigrams.sort_values(by='count', ascending=True)
        # Convert the 'Bigram' column to strings
        common_trigrams_sorted['trigram'] = common_trigrams_sorted['trigram'].apply(str)

        fig = px.bar(common_trigrams_sorted, x='count', y='trigram', orientation='h', color='count', text = 'count',color_continuous_scale='Inferno')

        # Update the tickvals and ticktext to match the reversed order
        tickvals = list(range(len(common_trigrams_sorted['trigram'])))
        ticktext = common_trigrams_sorted['trigram']

        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)

        fig.update_layout(
            title='Top 40 Trigrams Frequency (Most to Least)',
            xaxis_title='Count',
            yaxis_title='Trigram',
            showlegend=False,
            autosize=True,
            width=800,
            height=800,
        )

        # Place text outside the bars
        fig.update_traces(textposition='outside')

        st.plotly_chart(fig)


    elif analysis_choice == "Character Analysis":
        st.header("Character Analysis")
        # Group by character and count the number of sentences for each character
        character_counts = current_df.groupby('Character')['clean_sentence'].count()
        # Sort characters by spoken lines in ascending order and take the top 30
        top_characters = character_counts.sort_values(ascending=True).tail(30)
        # Plot horizontal bar graph
        # Create a DataFrame
        top_characters_df = pd.DataFrame({'Character': top_characters.index, 'Number of Spoken Lines': top_characters.values})

        # Create the bar chart using Plotly Express
        fig = px.bar(top_characters_df, x='Number of Spoken Lines', y='Character',
                    orientation='h', color='Number of Spoken Lines',text = 'Number of Spoken Lines',color_continuous_scale='Inferno')

        # Customize the chart layout
        fig.update_layout(
            title='Top 20 Characters by Number of Spoken Lines',
            xaxis_title='Number of Spoken Lines',
            yaxis_title='Character',
            showlegend=False,
            autosize=True,
            width=800,
            height=800,
        )

        # Place text outside the bars
        fig.update_traces(textposition='outside')

        # Display the Plotly chart
        st.plotly_chart(fig)
