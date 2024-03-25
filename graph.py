import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CSV file
all_characters_df = pd.read_csv('all_characters.csv')

# Get unique book titles
unique_books = all_characters_df['book'].unique()

# Sort books by the number of characters
book_counts = all_characters_df['book'].value_counts()
sorted_books = book_counts.index.tolist()

# Generate colors based on a color palette with the number of unique books
color_palette = sns.color_palette("rocket", len(unique_books))
book_colors = dict(zip(sorted_books, color_palette))

# Create bar plot for number of characters in each book
counts = all_characters_df['book'].value_counts()
ax = counts.plot(kind="bar", color=[book_colors[book] for book in counts.index])

# Set only numbers on the x-axis
plt.xticks(range(len(counts)), [])

# Add labels
plt.title("Number of Characters in Each Book")
plt.ylabel("Number of Characters")

# Add numbers on each bar
for i, value in enumerate(counts):
    plt.text(i, value, str(value), ha='center', va='bottom')

# Add legend matching the book names with bar colors
legend_handles = [plt.Rectangle((0,0),1,1, color=book_colors[book], ec="k") for book in counts.index]
plt.legend(legend_handles, counts.index, title='Books', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()  # Adjust layout to avoid overlapping
plt.show()
