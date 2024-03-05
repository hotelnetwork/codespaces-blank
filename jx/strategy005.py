from selenium import webdriver

# create a new Firefox browser instance
driver = webdriver.Chrome()

# navigate to the website
driver.get('https://olui2.fs.ml.com/Equities/OrderEntry.aspx?Symbol=TSLA')

# find the element
# element = driver.find_element_by_id('myElement')
# data = element.text
# print(data)

# close the browser

# Find all input elements
input_elements = driver.find_elements_by_tag_name("input")

# Get the values and names of all input elements
for input_element in input_elements:
    value = input_element.get_attribute("value")
    # name = input_element.get_attribute("name")
    print(f"Value: {value}")  # Name: {name}, Value: {value}")

driver.quit()
