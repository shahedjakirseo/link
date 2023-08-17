import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import ipywidgets as widgets
from IPython.display import display

# Load the data from the CSV file
csv_filename = "applavialinks.csv"
df = pd.read_csv(csv_filename)

# Preprocess the text data (URL, Title, and Description)
def preprocess_text(text):
    return text.lower()

df["URL"] = df["URL"].apply(preprocess_text)
df["Title"] = df["Title"].apply(preprocess_text)
df["Description"] = df["Description"].apply(preprocess_text)

# Combine Title and Description for better representation
df["Title_Desc"] = df["Title"] + " " + df["Description"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Title_Desc"])

# Create an Annoy index for fast approximate nearest neighbors search
annoy_index = AnnoyIndex(tfidf_matrix.shape[1], "angular")
for i in range(tfidf_matrix.shape[0]):
    annoy_index.add_item(i, tfidf_matrix[i].toarray()[0])
annoy_index.build(50)  # 50 trees for the index

# User input: link or title
input_widget = widgets.Text(
    value="Enter a link or a title",
    description="Input:",
    layout=widgets.Layout(width="80%"),
)
display(input_widget)

# Search button
search_button = widgets.Button(description="Search")
output_widget = widgets.Output()
display(search_button, output_widget)

def find_similar_links(query_vector, num_results=5):
    similar_indices = annoy_index.get_nns_by_vector(query_vector, num_results)
    return similar_indices

def search_button_clicked(b):
    with output_widget:
        output_widget.clear_output()
        user_input = input_widget.value.strip()
        user_vector = vectorizer.transform([user_input]).toarray()[0]
        similar_indices = find_similar_links(user_vector)
        print("Suggested similar links:")
        for idx in similar_indices:
            print(f"[{df.iloc[idx]['Title']}]({df.iloc[idx]['URL']}) - {df.iloc[idx]['Description']}")

search_button.on_click(search_button_clicked)