"use client";

import Image from "next/image";
import type { NextPage } from "next";
import { useAccount } from "wagmi";
import { Address } from "~~/components/scaffold-eth";

const Home: NextPage = () => {
  const { address: connectedAddress } = useAccount();

  return (
    <>
      <div className="">
        <div>
          <div className="hero bg-base-200 ">
            <div className="hero-content text-center ">
              <div className="px-5 items-center ">
                <div>
                  <h1 className="text-center">
                    <span className="block text-2xl mb-2">Welcome to</span>
                    <span className="block text-6xl font-bold">SMART PORTFOLIOS</span>
                  </h1>
                  <div className="flex justify-center items-center space-x-2 flex-col sm:flex-row">
                    <p className="my-2 font-medium">Connected Address:</p>
                    <Address address={connectedAddress} />
                  </div>
                  <p className="text-center  block text-2xl font-bold">
                    <h1> Explore Data-Driven DeFi Investing </h1>
                  </p>
                </div>
              </div>
            </div>
          </div>
          <div className=" items-center justify-center mb-10 p-10 grid grid-cols-2">
            <div>
              <h2 className="font-bold text-2xl">
                All transactions and decisions made by the model are visible and verifiable on the blockchain,
                generating trust for investors.
              </h2>
            </div>
            <div>
              <button className="btn btn-lg sm:btn-sm md:btn-md lg:btn-lg bg-slate-400 justify-end">
                Let Our ML Model Optimize Your Portfolio 24/7
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 place-items-center">
            <div className="card bg-base-100 w-96 shadow-xl">
              <figure>
                <Image src="/pc-1.jpeg" alt="Shoes" width={500} height={300} />
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
                <Image src="/rwa-1.jpeg" alt="Shoes" width={500} height={300} />
              </figure>
              <div className="card-body">
                <h2 className="card-title">Propose a new product</h2>
                <p>Experience the Power of create new investment products</p>
                <div className="card-actions justify-end">
                  <a href="/new-product">
                    <button className="btn btn-primary">Buy Now</button>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;
