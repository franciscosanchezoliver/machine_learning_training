- MLOps abbreviation for Machine Learning Operations. 
- Describe a set of practices to:
	- Design
	- Deploy
	- Maintain

-  Machine learning in production:
	- Continuously
	- Reliably
	- Efficiently

- Machine learning life cycle:
	1. Design
	2. Development
	3. Deployment

	- Design:
		- Problem definition & requirements
		- Exploratory data analysis
		- Implementation Design
	 
	-  Development:
		- Feature engineering
		- Experiment design
		- Model training and evaluation
	
	- Deployment:
		- Setup CI/CD pipeline
		- Deploy model
		- Monitoring

- Before making a machine learning model, we need to think about:
	- Business requirements
	- Added value of our model

- Combining data + requirements, find appropriate algorithm.

- MLOps originates from Development Operations, also called DevOps in short. 
	- DEV:
		- Plan
		- Create
		- Verify
		- Package
	- OPS:
		- Release
		- Configure
		- Monitor

- Traditional software development used to be slow because of the separation between development and operations teams
	- Development team: people who write the code
	- Operations team: people who deploy and support the code. 

- DevOps is the integration of both teams.
- Similar to DevOps is applied to software development, MLOps is applied to machine learning development.

- There are also best practices and tools for similar departments we can find in a IT Organization, such as ModelOps, DataOps, AIOps
![[Pasted image 20230909082746.png]]

- Each Ops originates from the same philosophy in DevOps, and focuses on continuous, reliable, and efficient development.

- ModelOps: can be seen as an extension of MLOps, with a set of practices primarily focused on the machine learning model.
- DataOps: focuses on best practices regarding data quality and analytics. Since data is part of machine learning, this overlaps with MLOps.
- AIOps: stands for Artificial Intelligence for IT Operations and is broader than just machine learning. In AIOps, analytics, big data, and machine learning are used to solved IT issues without human assistance or intervention.

- Benefits of MLOps:
	- Improve the speed of developing and delivering machine learning model.
	 - Processes also become more reliable and secure because of MLOps.
	 - It aims to bridge the gap between machine learning and operations teams, which enhance collaboration.

- Phases in MLOps:
	- 3 phases: design, development, deployment
	 - Iterative and cyclical process. Its common to go back and forth between phases.
	- It's important to constantly evaluate with stakeholders whether the machine learning project should be continued. Example: we can find in the design phase that we have limited data or that we can only apply the problem to a small group, this situation would reduce the added value and thus requires additional evaluation by stakeholders.
	- Gives a high level overview of how a machine learning project should be structured in order to deliver real/practical value.
	- It defines the roles that are required in each step.
	- We can apply certain practices and tools to each phase to optimize the life cycle.
 
- Design phase:
	- Focus on the design of the machine learning problem.
	- Define the context of the problem and determine the added value of using machine learning.
	- Gather business requirement, as well as establish key metrics through we can track the progress of the machine learning life cycle. 
	- Gather data to make sure the data quality is sufficient for developing a machine learning model.

- Development phase:
	- Focus on developing the machine learning problem.
	- Experimenting with a combination of data, algorithms and hyperparameters.
	- Train and evaluate one/many models in order to find the most suitable one.
	- Goal: get the most suitable machine learning model that is ready for deployment.


- Deployment phase:
	- Integrate the machine learning model into the business process. 
	- Might involve building a micro service from the machine learning model so that we can easily integrate the model into the business process.
	- Set up monitoring of the machine learning model. Set up alerts when we encounter data drift or when our model does not output a prediction anymore. 
		- Data drift occurs when our data changes, which impacts the machine learning model.
	 

-  Example of tasks per phase:
	- Design: 
		- Make an overview of the potential data sources that you need.
		- Talk to business manager to find out how accurate your machine learning model should be.
	- Development:
		- Do an experiment comparing two algorithms.
		- Use a combination of two features, for instance, the average per day, as input for your machine learning model.
	- Deployment:
		- Implement the machine learning model in the business process.
		- Do a weekly check of the machine learning model predictions.
 

Roles in MLOPS: https://campus.datacamp.com/courses/mlops-concepts/introduction-to-mlops?ex=7

- Roles involves in the machine learning life cycle.
	- For each task different roles are required.
	- 2 categories of roles: 
		- Business roles.
		- Technical roles.
  
- Business roles:
	-  2 roles:
		- Business stakeholder.
		- Subject matter expert.

- Business Stakeholder role:
	- Sometimes also referred to as the product owner.
	- Managerial staff. 
	- Making budget decisions.
	- Make sure that the machine learning project is aligned with the high-level vision of the company.
	- Involved throughout the life cycle. 
	- Define the business requirements during the design phase.
	- In the development phase they also see whether the initial results from the experiments are satisfactory
	- In the deployment phase, they examine whether the outcome of the life cycle is as expected.
 
- Subject matter expert: 
	- Has domain knowledge about the problem that we are trying to solve.
	- It's involved throughout the life cycle because they can assist the more technical roles with interpreting the data and results at each step.

- Technical roles:
	- 5 technical roles.
		- Data Engineer
		- Data Scientist
		- Software Engineer
		- ML Engineer
		- Backend Engineer. 

- Data Engineer
	- Responsible for:
		- Collect
		- Storing
		- Processing 
	- Tasks:
		- Check data quality
		- Include test such that the quality of is maintained throughout the process.
	- Involved with tasks that have to do with the data:
		- before training the model. 
		- during the model training. 
		- once the model is used in production.

- Data Scientist
	- Responsible for:
		- Data analysis
		- Model training
		- Model evaluation
			- This includes monitoring the model once it has been deployed to ensure that the model predictions are valid.
	- We can find the data scientist in all phases of the life cycle, but mostly during the development phase.

- Software Engineer:
	- Mainly involved during the deployment phase.
	- Write software to run the model. 
	- Deploy the model.
	- Monitor whether the model stays online once its deployed.
	- Make sure the code is written in accordance with common guidelines.
	- Since the deployment is an important part of the machine learning life cycle, the software engineer should also be included in the design phase.

- Machine Learning Engineer:
	- Relatively a new role.
	- Quite versatile. 
	- Have expertise over the entire machine learning life cycle.
	- Cross functional role that overlaps with the other technical roles.
	- Involved in all phases.
	- Knows how to:
		- Extract and store data
		- Develop a machine learning model
		- Deploy the machine learning model.

- Backend Engineer:
	- Mostly involved in setting up the cloud infrastructure to enable development and deployment of machine learning models. 
		- Example:
			- Database for storing the data. 
			- Computers to run the machine learning model. 



  


















