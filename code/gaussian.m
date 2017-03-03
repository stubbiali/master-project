% gaussian Evaluate a the Gaussian function 
% $f(x,\mu,\sigma) = \frac{1}{2 \pi \sigma^2} exp{\frac{-(x - \mu)^2}{2 \sigma^2}}$
% on a set of points.
%
% y = gaussian(x, mu, sigma)
% \param x      evaluation points
% \param mu     mean
% \param sigma  standard deviation
% \out   y      evaluations

function y = gaussian(x, mu, sigma)
    y = 1/sqrt(2*pi*sigma^2) * exp(-((x-mu).^2) ./ (2*sigma^2));
end