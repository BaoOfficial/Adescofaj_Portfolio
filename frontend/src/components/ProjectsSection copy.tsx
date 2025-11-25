import { ExternalLink, Github } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useState } from 'react';

const ProjectsSection = () => {
  const [showAll, setShowAll] = useState(false);
  
  const projects = [
  {
    title: "Customer Segmentation",
    description:
      "Performed customer segmentation using K-Means clustering to identify behavioral patterns and optimize marketing strategies. Includes data preprocessing, visualization, and interpretation.",
    image: "https://github.com/user-attachments/assets/d31f134b-d17c-442b-ba51-2432702f2f5f",
    technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "Jupyter Notebook"],
    github: "https://github.com/adescofaj/Customer-Segmentation",
    demo: null
  },
  {
    title: "Heart Disease Prediction",
    description:
      "Developed a predictive model using machine learning algorithms to assess heart disease risk based on clinical data. Focused on accuracy and interpretability using feature importance analysis.",
    image: "https://github.com/user-attachments/assets/ce52af51-3e6f-4f90-90f9-2ce1e09c4f06",
    technologies: ["Python", "Scikit-learn", "Logistic Regression", "Matplotlib", "NumPy"],
    github: "https://github.com/adescofaj/Heart-Disease-Prediction",
    demo: null
  },
  {
    title: "Python Coding Assistant",
    description:
      "An AI-powered coding assistant that provides explanations, debugging tips, and solutions to Python codes. Designed to enhance interactive learning for developers.",
    image: "https://github.com/user-attachments/assets/721ebc6a-67ce-431f-b7c1-00250856c61a",
    technologies: ["Python", "LangChain", "LangGraph", "OpenAI API", "FastAPI"],
    github: "https://github.com/adescofaj/Python-Coding-Assistant",
    demo: "https://drive.google.com/file/d/1qDmikowBdmonZ5CRaB4M-KW5BZcdGdvi/view?usp=drive_link"
  },
  {
    title: "Research Summarizer",
    description:
      "Built an NLP-based research summarizer that uses large language models to extract and condense academic papers into concise summaries for faster review.",
    image: "https://github.com/user-attachments/assets/8e511f8c-a10c-4027-8525-07e49cd37f5b",
    technologies: ["Python", "FastAPI", "Pydantic", "LangChain", "LangGraph", "OpenAI API", "PyPDF2"],
    github: "https://github.com/adescofaj/Research-Summarizer",
    demo: "https://drive.google.com/file/d/1aecouc1ZpdDfwNOJaCfp8RWVoocL1N85/view?usp=drive_link"
  },
  {
    title: "Tech Training Analytics",
    description:
      "Developed an end-to-end data analytics workflow for a corporate training platform, featuring data quality assessment, cleaning, database design, and business analytics.",
    image: "https://images.unsplash.com/photo-1531497865144-0464ef8fb9a9?w=500&h=300&fit=crop",
    technologies: ["MySQL", "MySQL Workbench (ERD generation)"],
    github: "https://github.com/adescofaj/Tech-Training-Analytics",
    demo: null
  }
];

  const displayedProjects = showAll ? projects : projects.slice(0, 6);

  return (
    <section id="projects" className="py-20 bg-background-secondary/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 scroll-reveal">
          <h2 className="text-4xl font-bold mb-4">
            Featured <span className="gradient-text">Projects</span>
          </h2>
          <div className="w-24 h-1 bg-gradient-primary mx-auto rounded-full mb-6"></div>
          <p className="text-xl text-foreground-muted max-w-2xl mx-auto">
            Explore my portfolio of data science projects showcasing machine learning, 
            AI, and analytics solutions across various domains.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {displayedProjects.map((project, index) => (
            <Card 
              key={project.title} 
              className="gradient-border bg-background-secondary border-0 overflow-hidden group scroll-reveal"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="relative overflow-hidden">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </div>
              
              <CardHeader>
                <CardTitle className="text-xl group-hover:text-accent-primary transition-colors duration-200">
                  {project.title}
                </CardTitle>
                <CardDescription className="text-foreground-muted">
                  {project.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="flex flex-wrap gap-1">
                  {project.technologies.map((tech) => (
                    <Badge 
                      key={tech} 
                      variant="outline" 
                      className="text-xs border-accent-primary/30 text-accent-primary"
                    >
                      {tech}
                    </Badge>
                  ))}
                </div>
                
                <div className="flex gap-2">
                  <a 
                    href={project.github} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="flex-1"
                  >
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="w-full border-accent-primary/30 hover:bg-accent-primary/10"
                    >
                      <Github className="w-4 h-4 mr-2" />
                      Code
                    </Button>
                  </a>
                  {project.demo && (
                    <a href={project.demo} target="_blank" rel="noopener noreferrer" className="flex-1">
                      <Button size="sm" className="w-full btn-gradient text-background">
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Demo
                      </Button>
                    </a>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        
        <div className="text-center mt-12">
          <Button 
            onClick={() => setShowAll(!showAll)}
            variant="outline"
            className="border-accent-primary/30 hover:bg-accent-primary/10 text-accent-primary"
          >
            {showAll ? 'Show Less' : 'Show More'}
          </Button>
        </div>
      </div>
    </section>
  );
};

export default ProjectsSection;