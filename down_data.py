from selenium import webdriver
import re
from lxml import etree
import time
import json

"""
The main task is to download data from the web page, and when the data is successfully downloaded, it will be saved locally
The second time you open the app, you can load the data locally instead of downloading it over the web
"""

# Accepts a data_list parameter. And keep the data of the downloaded web page to the data_list.
def get_house_data(data_list):
    ''' completes scraping data from the webpage, and when the data is successfully downloaded, it will be saved locally
    ----------
    Returns
    -------
    data_list: list
        A list contains apartment information     
           
    ''' 
    start_len = len(data_list) 
    print(data_list, start_len) # Prints the contents of the list and the length of the list.
    browse = webdriver.Chrome() # Create a browser object.
    browse.implicitly_wait(30) # Set the browser's automatic wait time, mainly to wait for the page to finish loading before starting to fetch data.
    browse.maximize_window() # Maximize the browser's window.
    for x in range(1, 20): # 
        print(f'Downloading page {x}......') 
        browse.get(rf'https://www.apartments.com/ann-arbor-mi/{x}/') 
        js = "window.scrollTo(0,888)"
        browse.execute_script(js) 
        time.sleep(5) # Set the program to sleep for five seconds. Prevent getting data too quickly. Seizure of IP
        text = browse.page_source  # Get the source code of the browser's web page. The type at this moment is str
        html = etree.HTML(text) 
        # Use regular expressions to look for the apartment information
        title_ = re.findall('class="js-placardTitle title">(.*?)</span', text, re.S)
        sub_title_ = re.findall('address js-url" title=".*?">(.*?)\d+</div', text, re.S)
        # Use the path of HTML to find subheadings.
        sub_title = html.xpath('//a/p[@class="property-address js-url"][1]/text()')
        sub_title2 = html.xpath('//a/p[@class="property-address js-url"][2]/text()')
        al = [x + ', ' + y for x, y in zip(sub_title, sub_title2)]
        sub_title_.extend(al)
        url_ = re.findall('data-listingid=".*?" data-url="(.*?)" data-streetaddress', text, re.S)
        pricing_ = re.findall('property-pricing">(.*?)</p>|rents">(.*?)</span>', text, re.S)
        beds_ = re.findall('property-beds">(.*?)</p>', text, re.S)

        for title, sub_title, url, pricing, beds in zip(title_, sub_title_, url_, pricing_, beds_):
            data_dict = {
                'title': title,
                'sub_title': sub_title,
                'url': url,
                'pricing': ''.join(pricing),
                'beds': beds.strip(),
            }
            data_list.append(data_dict)

# Exit the browser.
    browse.quit()
    # The program sleeps for 1 second
    time.sleep(1)
    print('Download completed, saving......')
    # Because of the messy data found during use,  a filter is added here if the length of each house dictionary subtitle is greater than two hundred. This information will be deleted.
    for x in data_list:
        if len(x['sub_title']) > 200:
            data_list.remove(x)

    # Dictionary of loop data_list. Deduplication is performed by converting the set function to a set.
    unique_list = list(set(frozenset(d.items()) for d in data_list))
    # Then restore the deduplicated data to a dictionary.
    unique_list = [dict(items) for items in unique_list]
    # Calculate the length by deduplicated list. Judge how much useful information we have crawled this time.
    print(f'A total of {len(unique_list) - start_len} pieces of information are captured')
    # Pass the information captured this time through the mode of w. Write directly to the apartment.json file
    with open('apartment.json', 'w') as f:
        json.dump(unique_list, f)
    print('save successfully......')



try:
    # Try reading local apartment.json. If the file exists, load the data into the variable data_list. And pass the list to the get_house_date function to get more data.
    with open('apartment.json', 'r') as f:
        data_list = json.load(f)
    get_house_data(data_list)

except FileNotFoundError as e:
    # If the local file is not found, the get_house_data function will be called directly and an empty list will be passed.
    print('Unable to find the local json cache, start downloading it from the web page.....')
    get_house_data([])








