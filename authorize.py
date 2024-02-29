from requests_oauthlib import OAuth1Session

# Define the URLs
request_token_url = 'https://api.etrade.com/oauth/request_token'
base_authorization_url = 'https://us.etrade.com/e/t/etws/authorize'

# Create an OAuth1Session instance
oauth = OAuth1Session('dcc206c501905ee19f6db67441dd4893', client_secret='20b6359de419e4bb0690a3505ff3369be254e52e2ca3a4ee790f36085d2ae8e7')

# Fetch a request token
fetch_response = oauth.fetch_request_token(request_token_url)

# Get the request token and secret
resource_owner_key = fetch_response.get('oauth_token')
resource_owner_secret = fetch_response.get('oauth_token_secret')

# Generate the authorization URL
authorization_url = oauth.authorization_url(base_authorization_url)

# Print the authorization URL so you can open it and authorize the app
print('Please go here and authorize:', authorization_url)
