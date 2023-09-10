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





