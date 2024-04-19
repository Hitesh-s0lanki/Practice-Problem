from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain.output_parsers import OutputFixingParser
from dotenv import load_dotenv
import os

load_dotenv()
    
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

class CandidateDetails(BaseModel):
    name: str = Field(description="Name of the candidate")
    email: str = Field(description="Email of the candidate")
    total_year_experience : float = Field(description="Total year of work experience")

class ListCandidate(BaseModel):
    candidates: List[CandidateDetails] =  Field("List of all Candidate data")


# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=ListCandidate)

prompt = PromptTemplate(
    template="""
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer.  
        <context>
        {context}
        </context>
        Question: {input}""",
    input_variables=["input", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

context = '''
candidate - 1 : ' +91 9930032332    nirajsolanki 729@gmail .com march 2022 - 2024                   other skills  attained    committee  member at sbr    co  sports  finance /  accounts  percentage remarks 9.40 cgpi - 64.50  scored 70+ in 2 subjects 72.75  scored 80 in acc.  and exemption in all 64.00  scored 85 in acc.  and exemption in 2 85.50  scored 95+ in accounts and maths 77  scored 90 in maths encouraged article to participate actively on taking session and share some key lesson for overall learning and growth.   session taken by me on technical and fundamental analysis in equity market  and dicussion on audit observation came  accross during audit  niraj solanki i article assistant  mumbai  maharashtra  work experience  sbr   co chartered accountants   mumbai   experiencepreparation of actual financials  statement  balance  p l  cash flow   as  per applicable financial reporting framework   that includes profit and loss reporting and financial reporting of 10+ private entity . also prepared provisional  and  projected financials. executed internal audit  and statutory audit  of 15+  private companies  llps   and insurance companies   starting from  determing the audit procedures to finalization including preparation of auditor s report along with caro reporting   limited review  and tax audit .  checking whether financial statements are prepared in accordance  with the compliance of the respective ind as .  practical working experience on  ind as 109  115  12  2  19. analysed financial  statements   performed ratios  and variance analysis  to identify crucial relationships   sales analysis  and projection   dupont analysis  confirm business understanding  and arrive at logical solution . experience of single handedly handling most of the audit  starting from risk assessment procedures   to applying  further audit procedures  assessing/identifying the risk of material misstatement and correcting the same  to the  finalising the audit. assisted  in statutory audit of sbi bank  branch and concurrent audit of federal bank   role played - verification of the  loan files   checking compliance with proper documentation before sanctioning of advances  filing lfar. assisted in framing standard operating procedure   sops  and risk and control matrix   racm  to improve operational  efficiency and controls. miscellaneous assignments  - due diligence of private company  incorporation of pvt ltd. co.  change in constitution of  llp  gst reconciliation  tan application  maitainted books of accounts of llp   application for ltds including  negotiation for the cash price.  technical  analytical and soft skills   microsoft office suite   financial analysis  time management  multi-tasking   interpersonal communication  problem solving  critical thinking  client handling and business knowledge   leadership  learning new skills  adaptable to new things. extra curricular activities 2021  nov  ca inter group ii icaiwon first prize in  sudoku solving  and chess playing competition in school. area of interest/ work looking for working in these area requires application and developing a skill set focused on analytical/critical thinking  and decision  making skills   which are the activites which interest me the most. by working in these area it  will allow me to  demonstrate my proficiency in accounting  maths   critical thinking  communication skills and  interest in finance. professional and academic qualification year degree institute/ collegefiling of 30+ tds return on quarterly basis including rectification of the same  40+ itr returns  5+ gst returns  5+ gstr  9  annual return  of entities inlcuding mca filings. 2019 hsc  class xii - comm  thakur college of science   comm 2017 ssc  class x  swami vivek. high school2020  nov  ca inter group i icai 2019  nov  ca foundation icai2022  june  bcom mumbai university  saraf col. '

candidate - 2 :'educationhiteshsolanki software engineer contact +91 9004713782 hiteshsolanki4623@gmail.com b/710  riddhi siddhi apt  mumbai  india portfolio mumbaiuniversity degree  bsc  in information technology gpa  9.9 / 10.02022 - 2025 2021 english  fluent  hindilanguagesskillsprofile responsible and motivated student ready to apply education in the workplace. offers excellent technical abilities with software and applications  ability to handle challenging work  and excellent time management skills. motivated student seeking internship in software engineering to gain hands-on experience. outgoing and friendly with strong drive to succeed. machine learning and  deep learning web scrapingstrong understanding of machine learning concepts and algorithms  including supervised learning  unsupervised learning  and reinforcement learning. proficient in feature engineering  model evaluation  and hyperparameter tuning techniques to optimize model performance. experienced in implementing machine learning algorithms using tensorflow  scikit-learn  and other libraries to solve real-world problems in domains such as natural language processing  computer vision  and recommendation systems. expertise in deep learning architectures such as convolutional neural networks  cnns   recurrent neural networks  rnns   and deep autoencoders.skilled in advanced deep learning techniques like transfer learning  generative adversarial networks  gans   and sequence-to- sequence models.experienced in applying deep learning to tasks like image classification  object detection  text generation  and time series forecasting comprehensive understanding of both frontend and backend development technologies. proficient in utilizing redis as a caching layer to improve the performance and scalability of web applications expertise in designing and developing restful and graphql apis to facilitate seamless communication between frontend and backend systems. proficient in building efficient and scalable web applications using next.js  a popular react framework.experienced in server-side rendering  ssr   static site generation  ssg   and client-side rendering  csr  with next.js.skilled in optimizing next.js applications for performance and seo. strong proficiency in data modeling  querying  and database administrationproficient in web scraping techniques using libraries such as beautifulsoup  scrapy  or selenium. skilled in extracting data from various sources on the web  including static html pages  dynamic javascript-rendered content  and apis. experienced in parsing and processing scraped data into structured formats like csv  json  or databases for further analysis or integration into machine learning pipelines.jai hind college  mumbai xii - maharashtra board   82.33  seth juggulal poddar academy  mumbai2019 x - icse 85  full stack  web and app  development projects codetech   https //codetech- new.vercel.app thread-clone  https //thread- gamma.vercel.app/ ai-saas   https //ai-saas-ruby- theta.vercel.app/ white board   https //miro-app- psi.vercel.app/ e-commerce app  https //github.com/hitesh- s0lanki/amazon-flutterleetcodegithublinkedln languages and tools c  c++  java  python  javascript  web assembly  typescript  bash   r'
'''


new_parser = OutputFixingParser.from_llm(parser=parser, llm = OpenAI())

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model | new_parser

# Invoke the prompt template and model
output = prompt_and_model.invoke({"context": context, "input": "Extract the name,email,total number of work experience of each candidates from the provided contexts.If experience is not there then 0, don't include the education year"})

print(output['candidates'])