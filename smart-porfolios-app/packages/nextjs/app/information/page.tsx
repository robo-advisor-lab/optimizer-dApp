import React from "react";

const page = () => {
  return (
    <div>
      <div className="hero bg-base-200 ">
        <div className="hero-content text-center">
          <div className="max-w-md">
            <h1 className="text-5xl font-bold">Smart Portfolios</h1>
            <p className="py-6 font-bold">
              Invest with confidence. Our BTC/ETH portfolio operates entirely on the blockchain, guaranteeing the
              traceability of your funds
            </p>
          </div>
        </div>
      </div>
      <div className=" items-center justify-center">
        <h2 className="font-bold">
          Transparency and Auditability: By operating on-chain, all transactions and decisions made by the model are
          visible and verifiable on the blockchain, generating trust for investors.
        </h2>
        <h2 className="font-sans">
          Data-Driven Decision Making: The use of AI models and on-chain data allows for objective decision-making based
          on real market information, eliminating emotional biases.
        </h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 place-items-center">
        <div className="card bg-base-100 w-96 shadow-xl">
          <figure>
            <img src="/pc-1.jpeg" alt="Shoes" />
          </figure>
          <div className="card-body">
            <h2 className="card-title">BTC/ETH</h2>
            <p>For traditional and institutional investors</p>
            <div className="card-actions justify-end">
              <a href="/products">
                <button className="btn btn-primary">Buy Now</button>
              </a>
            </div>
          </div>
        </div>

        <div className="card bg-base-100 w-96 shadow-xl">
          <figure>
            <img src="/bt-1.jpeg" alt="Shoes" />
          </figure>
          <div className="card-body">
            <h2 className="card-title">More</h2>
            <p>Create a new investment product</p>
            <div className="card-actions justify-end">
              <button className="btn btn-primary">Buy Now</button>
            </div>
          </div>
        </div>

        <div className="card bg-base-100 w-96 shadow-xl">
          <figure>
            <img src="/rwa-1.jpeg" alt="Shoes" />
          </figure>
          <div className="card-body">
            <h2 className="card-title">RWA ETF on-chain!</h2>
            <p>Experience the Power of RWA on-chain</p>
            <div className="card-actions justify-end">
              <button className="btn btn-primary">Buy Now</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default page;
