import requests

url = "https://text-sentiment.p.rapidapi.com/analyze"

payload = "text=I%20am%20very%20happy%20but%20sometimes%20not"
headers = {
    'content-type': "application/x-www-form-urlencoded",
    'x-rapidapi-key': "8f063108cfmsh3aa100a3fcfbaacp154179jsnb2004b15c7fc",
    'x-rapidapi-host': "text-sentiment.p.rapidapi.com"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)