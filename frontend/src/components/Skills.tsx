import React, { FC } from 'react';
import { Code2, Database, Brain, BarChart3, Cloud, Wrench, LucideIcon } from 'lucide-react';

interface SkillCardProps {
  icon: LucideIcon;
  title: string;
  skills: string[];
}

interface ProgressBarProps {
  name: string;
  percentage: number;
}

interface SkillCategory {
  icon: LucideIcon;
  title: string;
  skills: string[];
}

interface TechnicalSkill {
  name: string;
  percentage: number;
}

const SkillCard: FC<SkillCardProps> = ({ icon: Icon, title, skills }) => {
  return (
    <div className="bg-background-secondary border border-accent-primary/20 rounded-lg p-6 hover:border-accent-primary/50 transition-all duration-300">
      <div className="flex items-center gap-3 mb-4">
        <Icon className="w-6 h-6 text-accent-primary" />
        <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      </div>
      <ul className="space-y-2">
        {skills.map((skill, index) => (
          <li key={index} className="flex items-center gap-2 text-foreground-muted">
            <span className="w-1.5 h-1.5 bg-accent-primary rounded-full"></span>
            {skill}
          </li>
        ))}
      </ul>
    </div>
  );
};

const ProgressBar: FC<ProgressBarProps> = ({ name, percentage }) => {
  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-2">
        <span className="text-foreground font-medium">{name}</span>
        <span className="text-accent-primary font-semibold">{percentage}%</span>
      </div>
      <div className="w-full bg-background-secondary rounded-full h-2">
        <div 
          className="bg-gradient-to-r from-accent-primary to-accent-primary/60 h-2 rounded-full transition-all duration-1000"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};

const Skills: FC = () => {
  const skillCategories: SkillCategory[] = [
    {
      icon: Code2,
      title: "Programming Languages",
      skills: ["Python", "SQL", "JavaScript"]
    },
    {
      icon: Brain,
      title: "Machine Learning",
      skills: ["Supervised Learning", "Unsupervised Learning", "Deep Learning"]
    },
    {
      icon: Database,
      title: "Data Management",
      skills: ["PostgreSQL", "MongoDB", "MySQL"]
    },
    {
      icon: BarChart3,
      title: "Data Visualization",
      skills: ["Power BI", "Matplotlib", "Seaborn"]
    },
    {
      icon: Cloud,
      title: "Cloud Technologies",
      skills: ["AWS", "Azure", "Docker"]
    },
    {
      icon: Wrench,
      title: "Tools & Frameworks",
      skills: ["Git", "FastAPI", "LangChain"]
    }
  ];

  const technicalSkills: TechnicalSkill[] = [
    { name: "Python", percentage: 95 },
    { name: "Machine Learning", percentage: 90 },
    { name: "SQL", percentage: 85 },
    { name: "Generative AI", percentage: 85 },
    { name: "Data Visualization", percentage: 90 },
    { name: "PyTorch", percentage: 75 },
    { name: "Scikit-Learn", percentage: 92 },
    { name: "Power BI", percentage: 90 }
  ];

  return (
    <section id="skills" className="py-20 bg-background-secondary/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 scroll-reveal">
          <h2 className="text-4xl font-bold mb-4">
            Skills & <span className="gradient-text">Technologies</span>
          </h2>
          <div className="w-24 h-1 bg-gradient-primary mx-auto rounded-full mb-6"></div>
          <p className="text-xl text-foreground-muted max-w-2xl mx-auto">
            A comprehensive toolkit honed over years of experience in data science, 
            machine learning, and analytics projects.
          </p>
        </div>

        {/* Skill Categories Grid */}
        <div className="mb-20">
          <h3 className="text-2xl font-bold mb-8 text-center scroll-reveal">Expertise Areas</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {skillCategories.map((category, index) => (
              <div key={index} className="scroll-reveal" style={{ animationDelay: `${index * 100}ms` }}>
                <SkillCard 
                  icon={category.icon}
                  title={category.title}
                  skills={category.skills}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Technical Proficiency */}
        <div className="bg-background-secondary border border-accent-primary/20 rounded-lg p-8">
          <h3 className="text-2xl font-bold mb-8 text-center scroll-reveal">Technical Proficiency</h3>
          <div className="grid md:grid-cols-2 gap-x-8">
            {technicalSkills.map((skill, index) => (
              <div key={index} className="scroll-reveal" style={{ animationDelay: `${index * 50}ms` }}>
                <ProgressBar name={skill.name} percentage={skill.percentage} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;