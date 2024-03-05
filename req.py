import requests
from bs4 import BeautifulSoup

# Fetch the webpage
response = requests.get('https://finance.yahoo.com/quote/TSLA')

# Parse the webpage with bs4
soup = BeautifulSoup(response.text, 'html.parser')

# Find the price on the page
price_tag = soup.find('span', {'class': 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)'})

# Print the price
if price_tag is not None:
    print(price_tag.text)
else:
    print("Could not find the price.")
