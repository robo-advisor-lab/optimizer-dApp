import React from "react";
import Image from "next/image";

const page = () => {
  return (
    <div>
      <div className="container mx-auto px-4 py-16">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="md:w-1/2 mb-8 md:mb-0">
            <h2 className="text-4xl font-bold mb-4">Unlock Your Financial Potential</h2>
            <p className="text-gray-700 leading-relaxed">
              Our investment product is designed to help you achieve your financial goals. With a diversified portfolio
              and expert management, we&apos;ll guide you towards long-term success.
            </p>
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-6">
              Learn More
            </button>
          </div>
          <div className="md:w-1/2">
            <div className="card bg-base-100 w-96 shadow-xl">
              <figure>
                <Image src="/bt-1.jpeg" alt="Investment Product" width={384} height={256} />
              </figure>
              <div className="card-body">
                <h2 className="card-title">Purpose a new product</h2>
                <p>For traditional and institutional investors</p>
                <div className="card-actions justify-end">
                  <a href="/">
                    <button className="btn btn-primary">Buy Now</button>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Additional sections can be added here for features, benefits, testimonials, etc. */}
        <div className="mt-16">{/* ... more content ... */}</div>
      </div>
    </div>
  );
};

export default page;
