import { Github, Linkedin, Twitter, Mail, MessageCircle, Facebook } from 'lucide-react';

const SidebarElements = () => {
  return (
    <>
      {/* Left Sidebar - Social Icons */}
      <div className="fixed left-6 bottom-0 z-40 hidden lg:flex flex-col items-center space-y-4">
        <div className="flex flex-col space-y-4">
          <a 
            href="https://github.com/adescofaj" 
            className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
            aria-label="GitHub"
          >
            <Github className="w-6 h-6" />
          </a>
          <a 
            href="https://www.linkedin.com/in/adescofaj/" 
            className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
            aria-label="LinkedIn"
          >
            <Linkedin className="w-6 h-6" />
          </a>
          <a 
            href="#" 
            className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
            aria-label="Twitter"
          >
            <Twitter className="w-6 h-6" />
          </a>
          {/* <a 
            href="#" 
            className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
            aria-label="Facebook"
          >
            <Facebook className="w-6 h-6" />
          </a> */}
          <a 
            href="mailto:adescofaj@gmail.com" 
            className="text-foreground-muted hover:text-accent-primary transition-all duration-200 hover:scale-110 hover:-translate-y-1"
            aria-label="Email"
          >
            <Mail className="w-6 h-6" />
          </a>
        </div>
        
        {/* Vertical line */}
        <div className="w-px h-24 bg-foreground-muted"></div>
      </div>
      
      {/* Right Sidebar - Message Text */}
      <div className="fixed right-6 bottom-0 z-40 hidden lg:flex flex-col items-center">
        <div className="flex flex-col items-center space-y-4">
          <div 
            className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 cursor-pointer"
            style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
          >
            {/* <span className="text-sm font-medium tracking-wider">send me a message 📩</span> */}
            <a 
              href="mailto:adescofaj@gmail.com" 
              className="text-foreground-muted hover:text-accent-primary transition-colors duration-200 cursor-pointer"
              style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
            >
              <span className="text-sm font-medium tracking-wider">
                send me a message 📩
              </span>
            </a>
          </div>
          
          {/* Vertical line */}
          <div className="w-px h-24 bg-foreground-muted"></div>
        </div>
      </div>
    </>
  );
};

export default SidebarElements;