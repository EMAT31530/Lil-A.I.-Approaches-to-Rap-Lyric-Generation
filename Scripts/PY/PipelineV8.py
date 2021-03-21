# Package Imports
import pandas as pd
from pathlib import Path  # The Path class

# System Packages
import os
import sys

# For the more advanced requests
from github import Github

# Getpass is garbage, this is better, faster and non-event driven
from easygui import passwordbox


# Functions
def _onehot_login(__PAT=''):
    # Currently vulnerable to code-injection terminal attacks.
    yes_synonyms = ["yes", "y", "yh", "1", "true"]

    # Ask for login details
    if __PAT == '':
        _pat = passwordbox("Enter your GitHub PAT", "Input")  # Treat PAT like a password
        if _pat is None:
            print('Password cancelled. Halting execution')
            sys.exit(1)
    else:
        _pat = __PAT
    # Login
    g = Github(_pat)
    del _pat  # Immediately delete password from cache/RAM
    user = g.get_user()

    # Branch selection
    main_branch_bool = input("Main Branch: Yes or No? ")

    if main_branch_bool.lower() in yes_synonyms:
        _branch = "master"
    else:
        _branch = "PROTOTYPE"

    # Search for the repository in the user directory.
    r_proj_clone = 0
    for repo in user.get_repos():
        if repo.name == "ai-group-project-Team-JMJM":
            r_proj_clone = repo
            break

    if type(r_proj_clone) == int:
        print("ai-group-project-Team-JMJM not found")
        sys.exit(1)

    # Delete logins and details
    del g
    del user
    del repo

    print("Project found!")
    # r_proj_clone is a read AND write project clone.
    # Must be deleted during run-time
    return r_proj_clone, _branch


def csv_decode_and_write(file__, path_):
    data = file__.decoded_content
    data = data.decode('utf-8')[1:]
    with open(path_, 'w') as writefile:
        writefile.write(data)
    data = data.splitlines()
    data_rows = []
    for count, word in enumerate(data):
        if count > 0:
            data_rows.append(word.split(','))
    data = pd.DataFrame(data_rows)
    data = data.to_numpy()
    return data


def _decode_and_write(path_, contents_):
    """
    Internal function for decoding and writing a .txt file from contents to path_
    :param path_: str
    :param contents_: directory str
    :return: 0
    """
    RAP_DATA = []
    for file_ in contents_:
        path = file_.path
        path = str(path)
        # Only choose the .txt files
        if path[-4:] == '.txt':
            # Append the Lyrics
            RAP_DATA.append(file_.decoded_content.decode("utf-8"))

    temp_path = Path(path_)
    if temp_path.is_file():
        if os.stat(path_).st_size == 0:
            write_bool2 = True
        else:
            total_length = sum([len(lyric) for lyric in RAP_DATA])
            if os.stat(path_).st_size != total_length:
                write_bool2 = False
            else:
                with open(path_, 'w'):
                    pass  # Cheeky way to clear the file
                write_bool2 = True
    else:
        write_bool2 = True

    if write_bool2:
        for lyric in RAP_DATA:
            try:
                with open(path_, 'w', encoding="utf-8") as writefile:
                    writefile.write(lyric)
            except:
                print("Error, file moved/deleted during write")
        print("{} is now up to date!".format(path_))
    else:
        print("{} is already up to date!".format(path_))

    return 0


def _check_write_type(project_clone_, branch_, write_type_):
    """
    Function for checking what to write, THEN writing it
    :param project_clone_: github clone
    :param branch_: branch str
    :param write_type_: str ['both', 'clean-only', 'unclean-only']
    :return:
    """
    if write_type_ == 'clean-only':
        print("Importing Github cleaned text files...")
        contents = project_clone_.get_contents("RapLyrics/CLEAN", ref=branch_)
        print('Github files imported')
        path_name_ = 'AllLyrics_clean.txt'
    elif write_type_ == 'unclean-only':
        print("Importing Github uncleaned text files...")
        contents = project_clone_.get_contents("RapLyrics/UNCLEAN", ref=branch_)
        print('Github files imported')
        path_name_ = 'AllLyrics_unclean.txt'
    elif write_type_ == 'both':
        print("Importing Github uncleaned text files...")
        contents = project_clone_.get_contents("RapLyrics/UNCLEAN", ref=branch_)
        path_name_ = 'AllLyrics_unclean.txt'
        print('Github files imported')

        # Write the lyrics to a .txt from the clone
        _decode_and_write(path_name_, contents)

        print("Importing Github cleaned text files...")
        contents = project_clone_.get_contents("RapLyrics/CLEAN", ref=branch_)
        print('Github files imported')
        path_name_ = 'AllLyrics_clean.txt'
    else:
        print("Importing data does not support type {}".format(write_type_))
        sys.exit(1)

    # Write the lyrics to a .txt from the clone
    _decode_and_write(path_name_, contents)

    return 0


def import_github(ui_enabled=False, write_type='both', __PAT=''):
    """
    Function for importing the github file.
    Always faster than using Git, since free Git -> Github must cannot make individual request without cloning.
    :param __PAT: PAT
    :param ui_enabled: bool
    :param write_type: str ['both', 'clean-only', 'unclean-only']
    output: None
    """

    project_clone, branch = 0, 0

    # Secure one-hot login
    if not ui_enabled:
        project_clone, branch = _onehot_login(__PAT=__PAT)
    else:
        pass
        print('UI not built yet...')
        sys.exit(1)

    if type(project_clone) == int or type(branch) == 0:
        print('Login authentication error')
        sys.exit(1)

    # Write the correct .txt files
    _check_write_type(project_clone, branch, write_type)

    print("All files now up to date!")

    # Delete github object for security
    del project_clone

    return 0


def update_github(write_bool=False, ui_enabled=False, write_type='both', __PAT=''):
    """
    Function for updating the github file, by cleaning the lyrics, optional write to txt file.
    :param ui_enabled: bool
    :param write_bool: bool
    :param write_type: str ['both', 'clean-only', 'unclean-only']
    :param __PAT: str PAT
    output: None
    """
    project_clone, branch = 0, 0

    # Secure one-hot login
    if not ui_enabled:
        project_clone, branch = _onehot_login(__PAT)

    if type(project_clone) == int or type(branch) == 0:
        print('Login authentication error')
        sys.exit(1)
    print("Importing editing csv files...")

    # Split the long string into a list of lines, then split by words, then put into a csv, then to numpy arr
    contents = project_clone.get_contents("RapLyrics/Other", ref=branch)
    censors, capitals = 0, 0  # Intialisation
    for counter, file_ in enumerate(contents):
        path = file_.path
        path = str(path)
        title_start = path.find('Other')
        title_len = path[title_start:].find('.')
        name = path[title_start + 6:title_start + title_len + 4]
        print("Writing file {} {}".format(counter, name))
        if name.lower() == "censors.csv":
            censors = csv_decode_and_write(file_, name)
        elif name.lower() == "capitals.csv":
            capitals = csv_decode_and_write(file_, name)
        else:
            _ = csv_decode_and_write(file_, name)
    print("All editing csv files are up to date!")

    print("Importing Github uncleaned text files...")
    contents = project_clone.get_contents("RapLyrics/UNCLEAN", ref=branch)

    RAP_DATA = []
    rap_lyric_names = []

    for counter, file_ in enumerate(contents):
        path = file_.path
        print("File {} ".format(counter + 1) + path)
        path = str(path)

        # Only choose the .txt files
        if path[-4:] == '.txt':
            # Append the name
            title_start = path.find('UNCLEAN')
            title_len = path[title_start:].find('.')
            name = path[title_start + 8:title_start + title_len]
            if name[-2:] == 'UC':
                name = name[:-2]
            rap_lyric_names.append(name)

            # Append the Lyrics
        RAP_DATA.append(file_.decoded_content.decode("utf-8"))

        # Remove the \ufeff at the beginning O(n)
    for count, lyric in enumerate(RAP_DATA):
        RAP_DATA[count] = lyric[1:]

    # Censor the profanities O(n*m + n*m2) m > m2 xor m2 > m
    if type(censors) != int:
        for count in range(len(RAP_DATA)):
            for i in range(len(censors[0:])):
                RAP_DATA[count] = RAP_DATA[count].replace(str(censors[i, 0]), str(censors[i, 1]))

    if type(censors) != int:
        for count in range(len(RAP_DATA)):
            for i in range(len(capitals[0:])):
                RAP_DATA[count] = RAP_DATA[count].replace(str(capitals[i, 0]), str(capitals[i, 1]))

    contents = project_clone.get_contents("RapLyrics/CLEAN", ref=branch)
    print('\nImporting Github cleaned text files...')
    cleaned_names = []
    for counter, file_ in enumerate(contents):
        path = file_.path
        path = str(path)
        print("File {} ".format(counter + 1) + path)
        # Only choose the .txt files
        if path[-4:] == '.txt':
            # Append the name
            title_start = path.find('CLEAN')
            title_len = path[title_start:].find('.')
            name = path[title_start + 6:title_start + title_len]
            if name[-2:] == 'CL':
                name = name[:-2]
            cleaned_names.append(name)

    # If the (now cleaned) rap_lyrics name is new (not in cleaned_names), then we want to create that as a new file
    # If the (now cleaned) rap_lyrics name is NOT new (not in cleaned_names), then we want to update the file
    print("Commiting files back to github...")
    for counter, new_name in enumerate(rap_lyric_names):
        if new_name in cleaned_names:
            duplicate = project_clone.get_contents("RapLyrics/CLEAN/{}CL.txt".format(new_name), ref=branch)
            project_clone.update_file("RapLyrics/CLEAN/{}CL.txt".format(new_name),
                                      "This was uploaded automatically via pipeline", RAP_DATA[counter], duplicate.sha,
                                      branch=branch)
        else:
            project_clone.create_file("RapLyrics/CLEAN/{}CL.txt".format(new_name),
                                      "This was uploaded automatically via pipeline", RAP_DATA[counter], branch=branch)
    print('Files commited. Process finished. You will NEED to import now, update does not work')
    if write_bool:
        _check_write_type(project_clone, branch, write_type)

    # Delete github objects for security
    del project_clone
    del contents


def clean(lyrics):
    rap_data = lyrics

    censors = pd.read_csv("censors.csv")
    capitals = pd.read_csv("capitals.csv")
    censors = censors.to_numpy()
    capitals = capitals.to_numpy()

    for i in range(len(censors)):
        rap_data = rap_data.replace(str(censors[i, 0]), str(censors[i, 1]))

    for i in range(len(capitals)):
        rap_data = rap_data.replace(str(capitals[i, 0]), str(capitals[i, 1]))

    return rap_data


# Source: GitHub User: "MikeMNelhams"
# Last updated: 06/03/2021

if __name__ == "__main__":
    __PAT_G = passwordbox("Enter your GitHub PAT", "Input")  # Treat PAT like a password
    # raps = clean("AllLyrics_clean.txt", __PAT=__PAT_G)
    # print(raps)
    import_github(__PAT=__PAT_G)
    # import_github(__PAT=__PAT_G)
    del __PAT_G
