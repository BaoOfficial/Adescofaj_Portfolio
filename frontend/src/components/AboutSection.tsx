import { Badge } from '@/components/ui/badge';

const AboutSection = () => {
  const skills = [
    'Python', 'SQL', 'Scikit-learn', 'Pandas', 'NumPy',
    'Matplotlib', 'Seaborn', 'Power BI', 'Jupyter', 'Git & Github', 'Apache Spark'
  ];

  return (
    <section id="about" className="py-20 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 scroll-reveal">
          <h2 className="text-4xl font-bold mb-4">
            About <span className="gradient-text">Me</span>
          </h2>
          <div className="w-24 h-1 bg-gradient-primary mx-auto rounded-full"></div>
        </div>
        
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="scroll-reveal">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-primary rounded-2xl blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
              <div className="relative bg-background-secondary rounded-2xl p-1">
                <img
                  src="https://github.com/user-attachments/assets/9952d5e9-d71f-4377-bbf7-8a0ffd20e4c1"
                  alt="Adescofaj"
                  className="w-full max-w-sm mx-auto rounded-xl object-cover aspect-square"
                />
              </div>
            </div>
          </div>
          
          <div className="space-y-6 scroll-reveal">
            <div>
              <h3 className="text-2xl font-semibold mb-4 gradient-text">
                Turning Data into Intelligence
              </h3>
              <div className="space-y-4 text-foreground-muted leading-relaxed">
                <p>
                  With over 6 years of experience as a Data Scientist, I have a proven record in developing and deploying machine learning models — including pricing optimization and end-to-end data science solutions. I specialize in using advanced statistical methods to drive data-driven decisions, reduce costs, and improve efficiency.
                </p>
                <p>
                  Passionate about applying AI and machine learning to real-world challenges, I focus on delivering insights that create business value. Outside of work, I contribute to open-source projects and stay engaged with the latest in AI research.
                </p>
              </div>
            </div>
            
            <div>
              <h4 className="text-xl font-semibold mb-4 text-accent-primary">
                🛠️ Technologies & Tools
              </h4>
              <div className="flex flex-wrap gap-2">
                {skills.map((skill) => (
                  <Badge 
                    key={skill} 
                    variant="secondary"
                    className="bg-background-tertiary text-foreground border border-accent-primary/20 hover:border-accent-primary/50 hover:bg-accent-primary/10 transition-colors duration-200"
                  >
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;