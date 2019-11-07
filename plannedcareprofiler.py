import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('W:/MFT/Workforce Profiles/Planned_Care.csv')
print("Pre-merge length: " + str(len(df)))
retirestats = pd.read_csv('W:/Retirement Vulnerability/now.csv')
df = df.merge(retirestats[['Pay_Number','Over50', 'This year','1-2 years', '2-3 years', '3-5 years', 'time_to_retire',
                           'Reg/Unreg']], on='Pay_Number', how='left')
print("Post-merge length: " + str(len(df)))
print(df.columns)
from reportlab.lib.enums import TA_CENTER

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image




def basestats(x, title):
    headcount = (len(x))
    depts_count = len(x['department'].unique())
    wte = round(sum(x['WTE']), 1)
    female = round(len(x[x['Sex'] == 'F']) / headcount * 100, 1)
    male = round(100 - female, 1)
    age = round(sum(x['Age']) / headcount, 1)
    unregistered = len(x[x['Pay_Band'].isin(['2', '3', '4'])])
    registered = headcount - unregistered
    print("Registered: " + str(registered), "Unregistered: " + str(unregistered))

    head_table = {'Headcount': str(headcount), 'WTE':str(wte), 'Female %': str(female)+"%", 'Average age':str(age),
                  'Number of Depts':str(depts_count)}
    table1 = pd.DataFrame(list(head_table.items()), columns = ['Metric', 'Quantity'])
    bands = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, '9':0,
             'Medical and Dental': 0}
    for j in x['Pay_Band'].unique():
        y = len(x[x['Pay_Band'] == j])
        bands[j] = y
    bands['M&D'] = bands.pop('Medical and Dental')


def agecounts():
    plt.figure(0)
    plt.style.use('seaborn')
    agecounts = pd.value_counts(df['Age'].values).sort_index()

    graph = df.plot(kind='bar', x='Age', y='WTE')
    plt.legend('')
    graph = agecounts.plot.bar(color = '#003087')
    plt.title('Planned Care - Age Demography')
    plt.ylabel('Headcount')
    plt.xlabel('Age')

    for label in graph.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)


    plt.savefig('W:/MFT/Workforce Profiles/ageprofile.png', dpi=300)
    plt.close()

#todo AREA graphs


def subDir1WTE():
    plt.style.use('ggplot')
    graph2 = df.groupby('Sub-Directorate 1')['WTE'].sum().plot(kind = 'barh', color = '#003087')
    #graph2 = subdir.plot.barh()

    plt.title('WTE by Sub-Directorate')
    plt.xlabel('WTE')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/subdir1.png', dpi=300)
    plt.close()

def jobFamWTE():

    plt.style.use('seaborn')
    graph3 = df.groupby('Job_Family')['WTE'].sum().plot(kind = 'barh', color = '#003087')

    plt.ylabel('Job Family')
    plt.xlabel('WTE')
    plt.title('WTE by Job Family')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/jobfam.png', dpi=300)
    plt.close()
#todo exclude training grades
#todo differences between partnership/acute for a doctor agewise
#todo composition of job family in partnerships/acute


def payBandWTE():
    plt.style.use('seaborn')
    graph4 = df.groupby('Pay_Band')['WTE'].sum().sort_index().plot(kind = 'barh', color = '#003087')
    plt.ylabel('Pay Band')
    plt.xlabel('WTE')
    plt.title('WTE by Pay Band')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/paybands.png', dpi=300)
    plt.close()

def retirementProj():
    print(df['This year'].value_counts())
    labels = ['Imminent', 'Within a year', '1-2 years', '2-3 years', '3-5 years', '5-10 years', '>10 years']
    bins = [-100, 0,1,2,3,5,10,100]


    df['ProjRet'] = pd.cut(df['time_to_retire'], bins=bins, labels=labels, right=False)


    overallRetirement = pd.value_counts(df['ProjRet'].values, sort=False)
    print(type(overallRetirement))
    #pivot_df = df.pivot(index='ProjRet', columns='Reg/Unreg')
    #print(pivot_df)
    plt.style.use('seaborn')
    plt.ylabel('Headcount')
    plt.xlabel('Projected Retirement')
    overallRetirement.plot(kind='bar', color = '#003087')

    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/retirement.png', dpi = 300)
    plt.close()
#todo Put in retirement projections for all the staff.
#todo specifically look at reg/unreg (stacked chart)

def absenceTypes():
    pass
    #todo Absences in the past month


#todo Number of depts, subdirectorates, etc in the paragraphs
#todo what is the average nurse?
#todo length of service (earliest date available)


def pdfbuilder(i):
    absenceTypes()
    retirementProj()

    agecounts()
    subDir1WTE()
    jobFamWTE()
    payBandWTE()

    doc = SimpleDocTemplate(r"w://MFT/Workforce Profiles/" + i + "-email.pdf", rightMargin=10, leftMargin=10,
                            topMargin=10, bottomMargin=10)
    #
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", alignment=TA_CENTER, fontSize=10)
               )
    nhslogo = r'W:/Python/Danny/nhsggclogo.png'
    ageprofimg = Image('W:/MFT/Workforce Profiles/ageprofile.png', 5.5 * inch, 3.7 * inch)
    logo = Image(nhslogo, 0.9 * inch, 0.9 * inch)
    logo.hAlign = 'LEFT'
    header = i + ' - Workforce Profile - October 2019'
    retirement = Image('W:/MFT/Workforce Profiles/retirement.png', 5.5 * inch, 3.7 * inch)
    subdir1 = Image('W:/MFT/Workforce Profiles/subdir1.png', 5.5 * inch, 3.7 * inch)
    jobfam = Image('W:/MFT/Workforce Profiles/jobfam.png', 5.5 * inch, 3.7 * inch)
    paybands = Image('W:/MFT/Workforce Profiles/paybands.png', 5.5 * inch, 3.7 * inch)
    placeholder = Image('W:/MFT/Workforce Profiles/placeholder.png', 3 * inch, 3 * inch)
    data = [[logo, header]]
    headcount = (len(df))
    totalwte = str(round(sum(df['WTE']), 1))
    female = str(round(len(df[df['Sex'] == 'F']) / headcount * 100, 1))


    depts_count = str(len(df['department'].unique()))
    agetext = Paragraph('The '+i+' workforce comprises of '+str(headcount)+ ' employees ('+ totalwte
                        +' WTE) and ' + depts_count + ' departments. The workforce is '+ female + ' percent female.',
                        styles['Justify'])
    subdir1text = Paragraph('This is some text that is designed to explain the sub-directorate 1 graph above'+
                      '. This paragraph can also include specific stats etc.', styles['Justify'])
    jobfamtext = Paragraph('Here are some words to talk about Job Families.', styles['Justify'])
    paybandstext = Paragraph('Pay band text goes here.', styles['Justify'])
    retirementtext = Paragraph('These retirement vulnerability figures are calculated from historical (3 year) leaver'
                               +' trends. Can be separated by registered/unregistered or any other attributes as desired.',
                               styles['Justify'])
    sickabstext = Paragraph('Details of sickness absence will go here. Sickness absence is likely to have interesting'
                            +' interactions with age. Data will be taken from SSTS and we should be able to provide'
                            +' a 12 month absence profile and display as desired', styles['Justify'])
    banktext = Paragraph('We can also provide monthly bank usage figures across the extract. This could assist '
                        +' workforce planning goals and find potential pain points.', styles['Justify'])
    drivtext = Paragraph('This will be populated with current driving distance to primary workplace'+
                        ', which we can calculate using postcodes and the Bing Maps api. '
                        +'This kind of information may be useful for assessing potential ward moves etc.',
                         styles['Justify'])

    headertable = Table(data)
    headertable.setStyle(TableStyle([('VALIGN', (1, 0), (1, 0), 'MIDDLE')]))
    Story.append(headertable)
    Story.append(agetext)
    Story.append(Spacer(1, 12))
    Story.append(ageprofimg)

    Story.append(Spacer(1, 12))
    Story.append(subdir1)
    Story.append(Spacer(1, 12))
    Story.append(subdir1text)
    Story.append(Spacer(1, 12))
    Story.append(jobfam)
    Story.append(Spacer(1, 12))
    Story.append(jobfamtext)
    Story.append(Spacer(1, 12))
    Story.append(paybands)
    Story.append(Spacer(1, 12))
    Story.append(paybandstext)
    Story.append(Spacer(1, 12))
    Story.append(retirement)
    Story.append(Spacer(1, 12))
    Story.append(retirementtext)
    Story.append(Spacer(1, 12))
    Story.append(placeholder)
    Story.append(sickabstext)
    Story.append(Spacer(1, 12))
    Story.append(placeholder)
    Story.append(Spacer(1, 12))
    Story.append(banktext)
    Story.append(Spacer(1, 12))
    Story.append(placeholder)
    Story.append(Spacer(1, 12))
    Story.append(drivtext)
    Story.append(Spacer(1, 12))
    doc.build(Story)

pdfbuilder("Planned Care")