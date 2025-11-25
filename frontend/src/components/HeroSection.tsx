import { useState, useEffect } from 'react';
import { ArrowDown, Github, Twitter, Linkedin, Mail, Facebook } from 'lucide-react';
import { Button } from '@/components/ui/button';

const HeroSection = () => {
  const [currentRole, setCurrentRole] = useState(0);
  const roles = ['a Data Scientist', 'a ML Engineer', 'an AI Enthusiast'];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentRole((prev) => (prev + 1) % roles.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Floating Background Shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="floating-shape absolute top-1/4 left-1/4 w-32 h-32 bg-accent-primary/10 rounded-full blur-xl"></div>
        <div className="floating-shape absolute top-1/3 right-1/4 w-48 h-48 bg-accent-secondary/10 rounded-full blur-xl"></div>
        <div className="floating-shape absolute bottom-1/4 left-1/3 w-40 h-40 bg-accent-glow/10 rounded-full blur-xl"></div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        <div className="space-y-8 animate-fade-in">
          <div className="space-y-4">
            <p className="text-accent-primary font-medium text-lg">👋 Hi, my name is</p>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold">
              <span className="gradient-text">Amirlahi Fajingbesi</span>
            </h1>
            <div className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-foreground-muted">
              I'm {' '}
              <span 
                key={currentRole}
                className="gradient-text animate-fade-in inline-block"
              >
                {roles[currentRole]}
              </span>
            </div>
          </div>
          
          <p className="text-xl text-foreground-muted max-w-2xl mx-auto leading-relaxed">
            Passionate about transforming data into actionable insights and building 
            intelligent systems that solve real-world problems through machine learning and AI.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button 
              size="lg"
              className="btn-gradient text-background font-semibold px-8 py-3 text-lg"
              onClick={() => document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' })}
            >
              View My Work
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              className="border-accent-primary text-accent-primary hover:bg-accent-primary hover:text-background px-8 py-3 text-lg"
              onClick={() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' })}
            >
              Get In Touch
            </Button>
          </div>
          
          <div className="flex justify-center space-x-6 pt-8">
            <a 
              href="https://github.com/adescofaj" 
              className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 hover:scale-110 transform"
            >
              <Github className="w-6 h-6" />
            </a>
            <a 
              href="https://www.linkedin.com/in/adescofaj/" 
              className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 hover:scale-110 transform"
            >
              <Linkedin className="w-6 h-6" />
            </a>
            {/* <a 
              href="#" 
              className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 hover:scale-110 transform"
            >
              <Facebook className="w-6 h-6" />
            </a> */}
            <a 
              href="#" 
              className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
              aria-label="Twitter"
            >
              <Twitter className="w-6 h-6" />
            </a>
            <a 
              href="mailto:adescofaj@gmail.com" 
              className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 hover:scale-110 transform"
            >
              <Mail className="w-6 h-6" />
            </a>
          </div>
        </div>
        
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <ArrowDown className="w-6 h-6 text-accent-primary" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;