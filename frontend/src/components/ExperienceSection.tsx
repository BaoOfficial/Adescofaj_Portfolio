import { Calendar, MapPin } from 'lucide-react';

const ExperienceSection = () => {
  const experiences = [
  {
    type: 'experience',
    title: 'Data Scientist',
    company: 'PlayMode Music',
    location: 'Bournemouth, United Kingdom',
    period: 'Aug 2020 - Present',
    description: [
      'Led the development of machine learning models for demand forecasting and pricing optimization, resulting in a 20% improvement in streaming forecasts and a 15% improvement in revenue growth',
      'Collaborated with cross-functional teams using JIRA and version control to enhance data processing pipelines, reducing processing time by 40%',
      'Executed detailed data analysis through regression techniques, identifying correlations between user demographics and purchasing behavior; findings were instrumental in reshaping product offerings to attract 40% more targeted customers',
      'Created predictive models to optimise resource allocation, resulting in a 15% increase in revenue',
      'Applied supervised and unsupervised learning algorithms to perform customer segmentation, resulting in a targeted marketing approach and a 20% increase in customer retention'
    ]
  },
  {
    type: 'experience',
    title: 'Junior Data Scientist',
    company: 'PlayMode Music',
    location: 'Bournemouth, United Kingdom',
    period: 'Oct 2019 - Aug 2020',
    description: [
      'Developed and implemented machine learning algorithms for customer churn prediction, resulting in a 25% decrease in churn rate',
      'Conducted comprehensive exploratory data analysis and feature engineering, improving model accuracy by 15%',
      'Collaborated with software engineers to deploy machine learning models into production systems, enhancing system efficiency by 20%',
      'Supported A/B testing initiatives and performed statistical analysis on marketing campaigns, driving a 20% increase in conversion rates',
      'Created visually appealing data visualisations and reports, facilitating data-driven decision-making by stakeholders'
    ]
  },
  {
    type: 'experience',
    title: 'Data Analyst',
    company: 'Hype Multi-Media',
    location: 'Bournemouth, United Kingdom',
    period: 'Nov 2016 - Oct 2019',
    description: [
      'Analysed customer data and identified patterns, driving a 10% improvement in customer engagement and a 5% increase in sales',
      'Engaged with diverse teams to build interactive visualisations and dashboards using Python and SQL, empowering data-driven decision-making across the organisation',
      'Conducted A/B testing experiments and performed statistical analysis, resulting in a 10% uplift in conversion rates for marketing campaigns',
      'Contributed to the development of predictive models for customer churn forecasting, leading to a 15% reduction in customer attrition rates',
      'Ensured data governance and compliance with privacy regulations through data governance initiatives, maintaining data integrity and privacy standards'
    ]
  },
  {
    type: 'education',
    title: 'MSc Data Science & Artificial Intelligence',
    company: 'Bournemouth University',
    location: 'Bournemouth, UK',
    period: 'Sep 2022 - Sep 2023',
    description: [
      'Master of Science in Data Science & Artificial Intelligence'
    ]
  },
  {
    type: 'education',
    title: 'BSc Statistics',
    company: 'Olabisi Onabanjo University',
    location: 'Ogun State, Nigeria',
    period: 'Sep 2006 - Aug 2012',
    description: [
      'Bachelor of Science in Statistics'
    ]
  }
];

  return (
    <section id="experience" className="py-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 scroll-reveal">
          <h2 className="text-4xl font-bold mb-4">
            Experience & <span className="gradient-text">Education</span>
          </h2>
          <div className="w-24 h-1 bg-gradient-primary mx-auto rounded-full"></div>
        </div>
        
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-4 md:left-1/2 md:transform md:-translate-x-px h-full w-0.5 bg-gradient-primary"></div>
          
          <div className="space-y-12">
            {experiences.map((exp, index) => (
              <div 
                key={index} 
                className={`relative flex items-center ${
                  index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
                }`}
              >
                {/* Timeline dot */}
                <div className="absolute left-4 md:left-1/2 md:transform md:-translate-x-1/2 w-4 h-4 bg-accent-primary rounded-full border-4 border-background shadow-glow z-10"></div>
                
                {/* Content */}
                <div className={`ml-12 md:ml-0 md:w-1/2 ${
                  index % 2 === 0 ? 'md:pr-12' : 'md:pl-12'
                } scroll-reveal`}>
                  <div className="gradient-border bg-background-secondary p-6 rounded-lg shadow-elegant">
                    <div className="flex items-center gap-2 mb-2">
                      <Calendar className="w-4 h-4 text-accent-primary" />
                      <span className="text-accent-primary font-medium">{exp.period}</span>
                    </div>
                    
                    <h3 className="text-xl font-semibold mb-1">{exp.title}</h3>
                    <div className="flex items-center gap-2 mb-4 text-foreground-muted">
                      <span className="font-medium">{exp.company}</span>
                      <span>•</span>
                      <div className="flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        <span className="text-sm">{exp.location}</span>
                      </div>
                    </div>
                    
                    <ul className="space-y-2 text-foreground-muted">
                      {exp.description.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="w-1.5 h-1.5 bg-accent-primary rounded-full mt-2 flex-shrink-0"></span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ExperienceSection;