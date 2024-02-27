from bs4 import BeautifulSoup
import requests

# Make a request to the website
r = requests.get("https://www.wsj.com/market-data/quotes/TSLA")
r.content

# Use the 'html.parser' to parse the page
soup = BeautifulSoup(r.content, 'html.parser')

# print(soup.find_all('h3', class_='WSJTheme--headline--7VCzo7Ay'))

# Print the price
# print(price)
# Use the prettify() function to look at the webpage content
print(soup.prettify())

# Use the find() method to get the content of a specific tag
title = soup.find('title')
print(title.string)