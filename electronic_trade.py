from requests_oauthlib import OAuth1Session
import json

# Obtain these from your application page
consumer_key = 'dcc206c501905ee19f6db67441dd4893'
consumer_secret = '20b6359de419e4bb0690a3505ff3369be254e52e2ca3a4ee790f36085d2ae8e7'

# Fetch a request token
fetch_response = oauth.fetch_request_token(request_token_url)

# Get the request token and secret
resource_owner_key = fetch_response.get('oauth_token')
resource_owner_secret = fetch_response.get('oauth_token_secret')

# # Obtain these by running through the 'obtain a request token' sequence
# resource_owner_key = 'your_resource_owner_key'
# resource_owner_secret = 'your_resource_owner_secret'

# Create an OAuth1 session
etrade = OAuth1Session(consumer_key,
                       client_secret=consumer_secret,
                       resource_owner_key=resource_owner_key,
                       resource_owner_secret=resource_owner_secret)

# Define the order
order = {
    "PlaceEquityOrder": {
        "EquityOrderRequest": {
            "accountId": "your_account_id",
            "symbol": "TSLA",
            "orderAction": "BUY",
            "clientOrderId": "1234",
            "priceType": "LIMIT",
            "limitPrice": 100.00,
            "quantity": 1,
            "orderTerm": "GOOD_FOR_DAY",
            "marketSession": "REGULAR"
        }
    }
}

# Send the order
response = etrade.post('https://api.etrade.com/v1/accounts/{accountId}/orders/place', data=json.dumps(order))

# Print the response
print(response.text)
