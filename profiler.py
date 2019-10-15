import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('W:/MFT/Workforce Profiles/Planned_Care.csv')
print("Pre-merge length: " + str(len(df)))
retirestats = pd.read_csv('W:/Retirement Vulnerability/now.csv')
df = df.merge(retirestats[['Pay_Number','Over50', 'This year','1-2 years', '2-3 years', '3-5 years']], on='Pay_Number',
              how='left')
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
    df['RetirementStatus'] = np.where(df['This year'] == 1, 'Imminent / Within 1 year',
                                      np.where(df['1-2 years'] == 1, '1-2 years',
                                               np.where(df['2-3 years'] == 1, '2-3 years',
                                                        np.where(df['3-5 years'] == 1, '3-5 years', ''))))
    overallRetirement = pd.value_counts(df['RetirementStatus'].values, sort=False)
    overallRetirement.plot(kind='bar', color = '#003087')
    plt.style.use('seaborn')
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
    data = [[logo, header]]
    agetext = Paragraph('Here is an expandable paragraph where we talk about demographics. We can put in some'+
                      ' key stats within the text automatically with Python inbuilt string formatting.', styles['Justify'])
    subdir1text = Paragraph('Similarly, this is some text that is designed to explain the sub-directorate 1 graph above'+
                      '. This paragraph can also include specific stats etc.', styles['Justify'])
    jobfamtext = Paragraph('Here are some words to talk about Job Families.', styles['Justify'])
    paybandstext = Paragraph('Pay band text goes here.', styles['Justify'])
    retirementtext = Paragraph('Here are some words about retirement vulnerability.', styles['Justify'])
    headertable = Table(data)
    headertable.setStyle(TableStyle([('VALIGN', (1, 0), (1, 0), 'MIDDLE')]))
    Story.append(headertable)
    Story.append(ageprofimg)
    Story.append(Spacer(1, 12))
    Story.append(agetext)
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
    doc.build(Story)

pdfbuilder("Planned Care")