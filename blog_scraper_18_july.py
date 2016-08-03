import numpy as np
import pandas as pd
import requests, time, sys, codecs, cStringIO
import unicodecsv as csv
from lxml import html

USER_AGENT = {'user-agent': 'NerdWallet dwh bot'}
XPATH = '//div[@property="articleBody"]/p/text()|//div[@property="articleBody"]/p/*/text()|//div[@property="articleBody"]/p/*/*/text()|//div[@property="articleBody"]/h3/text()|//div[@property="articleBody"]/h3/*/text()'
###
filename = "sites_to_scrape_v1.txt"
FINAL_FILE_NAME = "csat_blog_scraped_content_v1.csv"


def get_url_info(url):
	try:
		r = requests.get(url, headers = USER_AGENT, allow_redirects = True, timeout = (5, 5))
		status_code = r.status_code
		if (status_code == requests.codes.ok):
			tree = html.fromstring(r.content)
			content = [string.encode("UTF-8") for string in tree.xpath(XPATH)]
			if len(content) > 0 and not content[0].startswith("by:"): ## dealing with names
				content = " ".join(content[1:])
				return([url, r.status_code, r.url, content])
			else:
				return([url, r.status_code, r.url, None])
		else: 
			output.append([url, r.status_code, r.url, None])
	except requests.exceptions.ConnectionError:
		print("Connection Error for " + url)
		return([url])
	except requests.exceptions.Timeout:
		print(url + " timed out.")
		return([url])
	except requests.exceptions.TooManyRedirects:
		print(url + " had too many redirects.")
		return([url])
	except requests.exceptions.RequestException as e:
		print("Some other error with " + url)
		return([url]) 


def main():
	try:
		## Get the URLs to test
		df = pd.read_csv(sys.argv[1], delimiter = ',')  
		url_list = ['https://www.nerdwallet.com/' + str(value) for values in df[["page_path_tx"]].values.tolist() for value in values]

		## Test the URLs, one batch at a time
		output = []
		for url in url_list[0:1]:
			url_info = get_url_info(url)
			output.append(url_info)
			print(url)
			time.sleep(0)
		if len(sys.argv) < 3:
			final_file_name = FINAL_FILE_NAME
		else:
			final_file_name = sys.argv[2]
		with open(final_file_name, 'wb') as final_file:
			writer = csv.writer(final_file, lineterminator = "\n", quoting = csv.QUOTE_ALL)
			for line in output:
				if line != None:
					writer.writerow(line)
	except IOError:
		print("Cannot open " + sys.argv[1])
	except Exception as e:
		print("Something else went wrong. Try again.")


if __name__ == "__main__":
	if len(sys.argv) >= 2:
		main()
	else:
		print("Input the first argument as file to parse through, second (optional) as the name of output.")

