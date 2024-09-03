import React from "react";

const page = () => {
  return (
    <div>
      <div className="hero bg-base-200 min-h-screen">
        <div className="hero-content flex-col lg:flex-row">
          <img src="/i-5.jpeg" className="max-w-sm rounded-lg shadow-2xl" />
          <div>
            <h1 className="text-5xl font-bold">Optimize Your Portfolio Today</h1>
            <p className="py-6">"Join the DeFi Revolution"</p>
            <a href="/information">
              <button className="btn btn-primary">Get Started</button>
            </a>
          </div>
        </div>
      </div>
      <div className="p-11">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-28">
          {" "}
          {/* Ajusta la grilla para dispositivos móviles */}
          <div className="card bg-gray-500 w-full md:w-96 shadow-xl items-center">
            {" "}
            {/* Ajusta el ancho en pantallas pequeñas */}
            <figure className="px-10 pt-10"></figure>
            <div className="card-body items-center text-center">
              <a href="/">
                <button className="btn btn-lg mt-10 btn-glass mb-5">BTC/ETH</button>
              </a>
            </div>
          </div>
          <div className="card bg-gray-500 w-full md:w-96 shadow-xl">
            {" "}
            {/* Ajusta el ancho en pantallas pequeñas */}
            <figure className="px-10 pt-10"></figure>
            <div className="card-body items-center text-center">
              <a href="/">
                <button className="btn btn-lg mt-10 bg-glass btn-rounded mb-5">
                  <h2>RWA ETF on-chain</h2> {/* Considera usar un tamaño de texto más adecuado para móviles */}
                </button>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default page;
