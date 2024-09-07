// pages/index.js
import Head from "next/head";

export default function Home() {
  return (
    <div className="bg-gray-900 min-h-screen text-white">
      <div>
        <div className="presentacion">
          <div className="titulo p-10">
            <h1 className=" font-bold text-2xl ">BTC/ETH portfolio</h1>
          </div>
          <div className="informacion p-10">
            {/* Aquí va el contenido de la presentación */}
            <p>
              Maximize your crypto returns with our innovative portfolio. We combine the top two cryptocurrencies with
              machine learning for optimal daily rebalancing.
            </p>
          </div>
        </div>
      </div>{" "}
      {/* Fondo oscuro y altura mínima de pantalla */}
      <Head>
        <title>Mi Dashboard</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="container mx-auto p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Primera fila */}
          <div className="bg-gray-800 rounded-lg shadow p-4 col-span-3">
            <h1 className="font-bold text-ellipsis">Historical returns</h1>
          </div>
          <div className="bg-gray-800 rounded-lg shadow p-4  ">
            <h1>Summary</h1>
            <div className="grid grid-cols-2 ">
              <div
                className="radial-progress bg-cyan-500 text-primary-content border-4 border-cyan-500 w-48 h-48"
                role="progressbar"
                data-value="70"
              >
                70%
              </div>
              <div className="p-20">
                <p>BTC</p>
                <p>ETH</p>
              </div>
            </div>
          </div>

          {/* Segunda fila */}
          <div className="bg-gray-800 rounded-lg shadow p-4">
            <div className="stats shadow">
              <div className="stat">
                <div className="stat-figure text-primary">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    className="feather feather-check"
                  >
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                </div>
                <div className="stat-title">risk aversion</div>
                <div className="stat-value text-primary"></div>
                <div className="stat-desc">21% more than last month</div>

                <div className="rating">
                  <input type="radio" name="rating-1" className="mask mask-star" />
                  <input type="radio" name="rating-1" className="mask mask-star" defaultChecked />
                  <input type="radio" name="rating-1" className="mask mask-star" />
                  <input type="radio" name="rating-1" className="mask mask-star" />
                  <input type="radio" name="rating-1" className="mask mask-star" />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg shadow p-4">
            <div className="indicator">
              <span className="indicator-item badge badge-secondary"></span>
              <div className="bg-base-300 grid h-32 w-32 place-items-center">content</div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg shadow p-4 col-span-2 ">
            <progress className="progress w-56 bg-cyan-500 top-24" value={0} max="100"></progress>
          </div>
        </div>
      </main>
    </div>
  );
}
