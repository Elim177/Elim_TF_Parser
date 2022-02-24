import markdown

# Load the file into file_content
file_content = [ line for line in open('decode_wav.md') ]

# Files to concatenate
api_doc = open('decode_wav.md','a')
tut_doc = open('keras.classification.txt','r')
for line in file_content:
    # We search for the correct section
    section = ""
    if line.startswith("<div"):
        section = line.strip()

    # Once we arrive at the correct position, write the new entry
    if section == "<div/>" :
        api_doc.write(tut_doc.read())

api_doc.close()

markdown.markdownFromFile(
    input= 'decode_wav.md' ,
    output='decode_wav.html',
    encoding='utf8',
)
