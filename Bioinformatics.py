import pandas as pd 
import streamlit as st 
import altair as alt
from PIL import Image

## Title Page


st.write("""

# DNA Nucleotide Count Web App

This app counts nucleotide composition of query DNA.


***
""")

##Input Text Box

#st.sidebar.header('Enter DNA sequence')
st.header('Enter DNA sequence')

sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

#sequence = st.sidebar.textarea("Sequence input", sequence_input, height=250)
sequence = st.text_area("Sequence input", sequence_input, height = 250)
sequence = sequence.splitlines()
sequence
sequence = sequence[1:] # Skips the sequence name (first line)
sequence = ''.join(sequence) # Concatenates list to string
sequence

st.write("""
***
""")

## Prints the input DNA sequence
st.header('First 120 Nucleotides')
sequence[0:30]
sequence[30:60]
sequence[60:90]
sequence[90:120]


st.header('Complementary DNA Sequence')
#def comp_pair(seq):

base = []
for nuc in sequence[0:120]:
	if nuc == "A":
		base.append ("T")
	if nuc == "T":
		base.append ("A")
	if nuc == "G":
		base.append ("C")
	elif nuc == "C":
		base.append ("G")
base = ''.join(base)
base[0:30]
base[30:60]
base[60:90]
base[90:120]

	


#C = comp_pair(sequence)
#C
		
	

## DNA nucleotide count
st.header('DNA Nucleotide Count')

### 1. Print dictionary
st.subheader('1. Print dictionary')
def DNA_nucleotide_count(seq):
	d = dict([
			('A', seq.count('A')),
			('T', seq.count('T')),
			('G', seq.count('G')),
			('C', seq.count('C'))
			])
	return d

X = DNA_nucleotide_count(sequence)

#X_label = list(X)
#X_values = list(X.values())

X 

# 2. Print test
st.subheader('2. Print text')
st.write('There are  ' + str(X['A']) + ' adenine (A)')
st.write('There are  ' + str(X['T']) + ' thymine (T)')
st.write('There are  ' + str(X['G']) + ' guanine (G)')
st.write('There are  ' + str(X['C']) + ' cytosine (C)')

# 3. Display DataFrame
st.subheader('3. DF Display')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'nucleotide'})
st.write(df)

# 4. Display Bar Chart using Altair
st.subheader('4. Bar Chart Display')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)
p = p.properties(
    width=alt.Step(80)  # controls width of bar.
)
st.write(p)
