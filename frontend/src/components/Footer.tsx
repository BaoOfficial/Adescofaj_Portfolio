import { Heart } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="py-8 border-t border-accent-primary/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="w-24 h-0.5 bg-gradient-primary mx-auto mb-6"></div>
          <p className="text-foreground-muted flex items-center justify-center gap-2">
            Built with React
          </p>
          <p className="text-sm text-foreground-muted mt-2">
            © 2025 Amirlahi Ademola Fajingbesi. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;