import { useEffect } from 'react';
import Navigation from '@/components/Navigation';
import HeroSection from '@/components/HeroSection';
import AboutSection from '@/components/AboutSection';
import ProjectsSection from '@/components/ProjectsSection';
import ExperienceSection from '@/components/ExperienceSection';
import ContactSection from '@/components/ContactSection';
import Footer from '@/components/Footer';
import SidebarElements from '@/components/SidebarElements';
import Chatbot from '@/components/Chatbot';
import { useScrollReveal } from '@/hooks/useScrollReveal';
import Skills from '@/components/Skills';

const Index = () => {
  useScrollReveal();

  return (
    <div className="min-h-screen">
      <Navigation />
      <SidebarElements />
      
      <main>
        <HeroSection />
        <AboutSection />
        <ProjectsSection />
        <Skills />
        <ExperienceSection />
        <ContactSection />
      </main>
      
      <Footer />
      <Chatbot />
    </div>
  );
};

export default Index;
