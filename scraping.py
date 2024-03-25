import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt
import os
import logging

# TODO: If needed add time.sleep(3) if necessary

# Setup ChromeOptions
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--ignore-ssl-errors=yes')
chrome_options.add_argument('--ignore-certificate-errors')
# chrome_options.add_experimental_option("detach", True)

# Setup Chrome Driver
driver = webdriver.Chrome(options=chrome_options, service=ChromeService(ChromeDriverManager().install()))

# Page to get with Driver where all books are
page_url = "https://witcher.fandom.com/wiki/Category:Characters_in_the_stories"
driver.get(page_url)

# # if there is popup then click on button with text Accept
# driver.find_element(By.XPATH, '//buton[text()="Accept"]')

# Find book titles with class name of elements
book_category = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')
# print(book_category[0].text) # --> Book title

# Links for all books and name of all books
links = [link.get_attribute("href") for link in book_category]
books_name = [book.text for book in book_category]
print(links)
print(books_name)

# Create list with dictionary
books = []
for category in book_category:
    book_url = category.get_attribute("href")
    book_name = category.text
    books.append({ "book_name": book_name, "book_url": book_url })
print(books)


# Find all characters with class name of elements (for 1 book)
characters = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')

# Find all characters in all books
all_characters = []
for book in books:
    driver.get(book['book_url'])
    characters = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')
    for character in characters:
        all_characters.append({"book": book['book_name'], "character":character.text})
print(all_characters)

# Transform for better visualization into DataFrame with pandas and save data in csv
all_characters_df = pd.DataFrame(all_characters)
all_characters_df.to_csv('all_characters_books.csv', index=True)
print(all_characters_df)

# Create bar plot for number of characters in each book  
all_characters_df['book'].value_counts().plot(kind="bar")
# Adding labels
plt.title("Number of Characters in Each Book")
plt.xlabel("Book")
plt.ylabel("Number of Characters")
# Adding numbers on each bar

# Add numbers on each bar
for i, value in enumerate(all_characters_df['book'].value_counts()):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.show()
