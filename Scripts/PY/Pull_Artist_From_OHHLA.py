# Free lyrics: URL http://ohhla.com/all.html/robots.txt

from selenium import webdriver
from bs4 import BeautifulSoup
import lxml
import urllib3
import requests
import pandas as pd

import sys
import os
import warnings
from pathlib import Path
import csv


def smallest_length_index(array):
    length = 200
    f = 0
    for index in range(len(array)):
        if len(array[index]) < length:
            length = len(array[index])
            f = index
    return f


ohhla_url = "http://ohhla.com/all.html/robots.txt"
ohhla_url_default = "http://ohhla.com/"
allowed_letters = ['all.html#xx', 'all.html#a', 'all.html#b', 'all.html#c', 'all.html#d', 'all.html#e',
                   'all_two.html', 'all_three.html', 'all_four.html', 'all_five.html']

rapper_name = "snoop dogg"

rapper_name = rapper_name.lower()
rapper_name_parsed = rapper_name.replace(' ', '_')


# Make sure we haven't already checked for this rapper
if Path('OHHLA/{}'.format(rapper_name_parsed)).is_dir():
    pass
    # print('Rapper \'{}\' has already been web-scraped '
    #       'in directory \'{}\''.format(rapper_name, 'OHHLA/{}'.format(rapper_name_parsed)))
    # sys.exit(0)


if not Path('OHHLA/{}'.format(rapper_name_parsed)).is_dir():
    os.mkdir('OHHLA/{}'.format(rapper_name_parsed))

http = urllib3.PoolManager()
save_urls_path = 'OHHLA/{}/{}_urls.csv'.format(rapper_name_parsed, rapper_name_parsed)

if not Path(save_urls_path).is_file():
    driver = webdriver.Chrome("chromedriver")
    search_letter_dict = pd.read_csv('OHHLA/search_dict.csv', header=None).to_numpy()
    search_letter_dict = dict(zip(search_letter_dict[:, 0], search_letter_dict[:, 1]))
    search_url = ohhla_url_default + allowed_letters[search_letter_dict[rapper_name[0]]]
    print('Searching: {} for artist \'{}\''.format(search_url, rapper_name))

    # Open the search page for the rapper
    driver.get(search_url)
    content = driver.page_source
    soup = BeautifulSoup(content, features="html.parser")

    # - Possible list of URLS containing the rapper. We will always pick the shortest
    possible_rapper_urls = [a['href'] for a in soup.findAll('a', href=True, text=True) if rapper_name in a.text.lower()]
    print('Possible URLS found: ', possible_rapper_urls)

    rapper_songs_url = ohhla_url_default + possible_rapper_urls[smallest_length_index(possible_rapper_urls)]
    print('Compiling songs from best match: {}'.format(rapper_songs_url))

    # Open the songs page for the rapper (best match)
    driver.get(rapper_songs_url)
    content = driver.page_source
    soup = BeautifulSoup(content, features="html.parser")

    driver.quit()  # No longer needed

    # Select only the anchors within the righthandsides of tables that have hrefs
    anchors = [a for a in soup.findAll('a', href=True, text=True)]
    anchors = [a for a in anchors if a.parent.has_attr('align') and a.parent['align'] == 'left']

    # Make a phat list of these URLS and write to a csv file
    print('Writing all song URL extensions to \'{}\' for artist \'{}\''.format(save_urls_path, rapper_name))

    with open(save_urls_path, 'w') as read_file:
        read_file.write('song_name,url_extension\n')

    with open(save_urls_path, 'a', newline='') as read_file:
        writer = csv.writer(read_file)
        for a in anchors:
            writer.writerow([a.text, a['href']])

    del anchors


# Open the CSV, make a dictionary from it and loop through the values (URLs)
url_extensions = pd.read_csv(save_urls_path).to_numpy()[:, 1]

# Remove annoying boilerplate language and stuff that doesn't actually add to text.
# ~ custom delimeter looks cool lol
with open('OHHLA/remove_text.txt', 'r') as read_file:
    removals = read_file.read().split('~')

# Remove any lines that contain these sequences
with open('OHHLA/remove_text_line_killers.txt', 'r') as read_file:
    line_killers = read_file.read().split('~')

if '' in line_killers:
    print('\'\' found in \'remove_text_line_killers.txt\' please check newlines and delimiter is \'~\'')
    sys.exit(1)

save_path_data = 'OHHLA/{}/{}_data.txt'.format(rapper_name_parsed, rapper_name_parsed)
print('Aggregating data to {}'.format(save_path_data))

with open(save_path_data, 'w') as read_file:
    read_file.close()

with open(save_path_data, 'a') as read_file:
    num_urls = len(url_extensions)
    for counter, url in enumerate(url_extensions):
        target_url = ohhla_url_default + url
        print('Currently downloading file {} of {}. {}'.format(counter, num_urls, target_url))

        # Try a safe ping using http.request from urllib3
        http.request('GET', target_url)

        # Less safe ping that we just want data from
        response = requests.get(target_url)
        soup = BeautifulSoup(response.text, features="lxml")
        soup = soup.get_text()

        # Remove the double \n
        soup = soup.replace('\n\n', '')

        # Remove the tabs \t
        soup = soup.replace('\t', '')

        # Remove the fucking annoying BOMB encoding chars
        soup = soup.replace('\\ufffd', '') + '\n'

        # Remove the random prefix information
        for annoying_text in removals:
            soup = soup.replace(annoying_text[1:], '')

        # Remove these annoying spatial sequences
        soup = soup.replace('  ', '')

        try:
            read_file.write(soup)
        except UnicodeEncodeError:
            warnings.warn('Error writing file {} of {}. {}'.format(counter, num_urls, target_url))
            warnings.warn('Check the encoding on file: {} is \'utf-8\''.format(target_url))

# https://stackoverflow.com/questions/11968998/remove-lines-that-contain-certain-string
# Now we want to temporarily write a file of all the info without line_killers
with open(save_path_data, 'r') as read_file, open('OHHLA/temporary_file.txt', 'w') as temp_file:
    for line in read_file:
        if not any(line_killer in line for line_killer in line_killers):
            temp_file.write(line)

# Move back to read_file
with open(save_path_data, 'w') as read_file, open('OHHLA/temporary_file.txt', 'r') as temp_file:
    temp_data = temp_file.read()
    read_file.write(temp_data)

# Delete the temporary_file
os.remove('OHHLA/temporary_file.txt')

print('Process finished. Information saved to {}'.format(save_path_data))
